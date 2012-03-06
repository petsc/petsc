/* The code is active only when the flag PETSC_USE_PTHREAD is set */

#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

int*           pVal_lockfree;

typedef void* (*pfunc)(void*);

/* lock-free data structure */
typedef struct {
  pfunc *funcArr;
  void** pdata;
  int *my_job_status;
} sjob_lockfree;
sjob_lockfree job_lockfree = {NULL,NULL,0};

#define PetscAtomicCompareandSwap(ptr, oldval, newval) (__sync_bool_compare_and_swap(ptr,oldval,newval))

void* FuncFinish_LockFree(void* arg) {
  PetscAtomicCompareandSwap(&PetscThreadGo,PETSC_TRUE,PETSC_FALSE);
  return(0);
}

/* 
  ----------------------------
     'LockFree' Thread Functions 
  ----------------------------
*/
void* PetscThreadFunc_LockFree(void* arg) 
{
  int iVal;

  iVal = *(int*)arg;
  pthread_setspecific(rankkey,&threadranks[iVal+1]);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  DoCoreAffinity();
#endif

  /* Spin loop */
  while(PetscThreadGo) {
    if(job_lockfree.my_job_status[iVal] == 0) {
      if(job_lockfree.funcArr[iVal+PetscMainThreadShareWork]) {
	job_lockfree.funcArr[iVal+PetscMainThreadShareWork](job_lockfree.pdata[iVal+PetscMainThreadShareWork]);
      }
      PetscAtomicCompareandSwap(&job_lockfree.my_job_status[iVal],0,1);
    }
  }
  __sync_bool_compare_and_swap(&job_lockfree.my_job_status[iVal],0,1);

  pthread_setspecific(rankkey,NULL);
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationInitialize_LockFree"
PetscErrorCode PetscThreadsSynchronizationInitialize_LockFree(PetscInt N)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pVal_lockfree = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  job_lockfree.funcArr = (pfunc*)malloc((N+PetscMainThreadShareWork)*sizeof(pfunc));
  job_lockfree.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  job_lockfree.my_job_status = (int*)malloc(N*sizeof(int));

  threadranks[0] = 0; /* rank of main thread */
  pthread_setspecific(rankkey,&threadranks[0]);
  /* Create threads */
  for(i=0; i<N; i++) {
    pVal_lockfree[i] = i;
    threadranks[i+1] = i+1;
    job_lockfree.my_job_status[i] = 1;
    job_lockfree.funcArr[i+PetscMainThreadShareWork] = NULL;
    job_lockfree.pdata[i+PetscMainThreadShareWork] = NULL;
    ierr = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_lockfree[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationFinalize_LockFree"
PetscErrorCode PetscThreadsSynchronizationFinalize_LockFree() 
{
  int i;
  void* jstatus;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Signal all threads to finish */
  PetscThreadsRunKernel(FuncFinish_LockFree,NULL,PetscMaxThreads,PETSC_NULL);

  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);
  }

  pthread_setspecific(rankkey,PETSC_NULL);
  free(job_lockfree.my_job_status);
  free(job_lockfree.funcArr);
  free(PetscThreadPoint);
  free(pVal_lockfree);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsWait_LockFree"
void* PetscThreadsWait_LockFree(void* arg) 
{
  int job_done,i;

  job_done = 0;
  /* Loop till all threads signal that they have done their job */
  while(!job_done) {
    for(i=0;i<PetscMaxThreads;i++)
      job_done += job_lockfree.my_job_status[i];
    if(job_done == PetscMaxThreads) job_done = 1;
    else job_done = 0;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_LockFree"
PetscErrorCode PetscThreadsRunKernel_LockFree(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) 
{
  int i,j,issetaffinity=0;

  PetscFunctionBegin;
  for(i=0;i<PetscMaxThreads;i++) {
    if(pFunc == FuncFinish_LockFree) {
      job_lockfree.funcArr[i+PetscMainThreadShareWork] = pFunc;
      job_lockfree.pdata[i+PetscMainThreadShareWork] = NULL;
    } else {
      issetaffinity=0;
      for(j=PetscMainThreadShareWork;j < n;j++) {
	if(cpu_affinity[j] == ThreadCoreAffinity[i]) {
	  job_lockfree.funcArr[i+PetscMainThreadShareWork] = pFunc;
	  job_lockfree.pdata[i+PetscMainThreadShareWork] = data[j];
	  issetaffinity=1;
	}
      }
      if(!issetaffinity) {
	job_lockfree.funcArr[i+PetscMainThreadShareWork] = NULL;
	job_lockfree.pdata[i+PetscMainThreadShareWork] = NULL;
      }
    }
    /* signal thread i to start the job */
    PetscAtomicCompareandSwap(&(job_lockfree.my_job_status[i]),1,0);
  }
  
  if(pFunc != FuncFinish_LockFree) {
    /* If the MainThreadShareWork flag is on then have the main thread also do the work */
    if(PetscMainThreadShareWork) {
      job_lockfree.funcArr[0] = pFunc;
      job_lockfree.pdata[0] = data[0];
      job_lockfree.funcArr[0](job_lockfree.pdata[0]);
    }
    /* Wait for all threads to finish their job */
    PetscThreadsWait(NULL);
  }

  PetscFunctionReturn(0);
}
