/* The code is active only when the flag PETSC_USE_PTHREAD is set */

#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

static PetscInt*           pVal_lockfree;

typedef void* (*pfunc)(void*);
/* lock-free data structure */
typedef struct {
  pfunc *funcArr;
  void** pdata;
  PetscInt *my_job_status;
} sjob_lockfree;

static sjob_lockfree job_lockfree = {NULL,NULL,0};

/* This struct holds information for PetscThreadsWait_LockFree */
static struct {
  PetscInt nthreads; /* Number of busy threads */
  PetscInt *list;    /* List of busy threads */
} busy_threads;

#if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 1050
#define PetscAtomicCompareandSwap(ptr, oldval, newval) (OSAtomicCompareAndSwapPtr(oldval,newval,ptr))
#elif defined(_MSC_VER)
#define PetscAtomicCompareandSwap(ptr, oldval, newval) (InterlockedCompareExchange(ptr,newval,oldval))
#elif (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
#define PetscAtomicCompareandSwap(ptr, oldval, newval) (__sync_bool_compare_and_swap(ptr,oldval,newval))
#else
#  error No maping for PetscAtomicCompareandSwap
#endif

static void* PetscThreadsFinish_LockFree(void* arg) {
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
  PetscInt iVal;

  iVal = *(PetscInt*)arg;
  pthread_setspecific(PetscThreadsRankKey,&PetscThreadsRank[iVal+1]);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscThreadsDoCoreAffinity();
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
  PetscAtomicCompareandSwap(&job_lockfree.my_job_status[iVal],0,1);

  pthread_setspecific(PetscThreadsRankKey,NULL);
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationInitialize_LockFree"
PetscErrorCode PetscThreadsSynchronizationInitialize_LockFree(PetscInt N)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(N*sizeof(PetscInt),&pVal_lockfree);CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(pthread_t),&PetscThreadPoint);CHKERRQ(ierr);
  ierr = PetscMalloc((N+PetscMainThreadShareWork)*sizeof(pfunc),&(job_lockfree.funcArr));CHKERRQ(ierr);
  ierr = PetscMalloc((N+PetscMainThreadShareWork)*sizeof(void*),&(job_lockfree.pdata));CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(PetscInt),&(job_lockfree.my_job_status));CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(PetscInt),&(busy_threads.list));CHKERRQ(ierr);

  PetscThreadsRank[0] = 0; /* rank of main thread */
  pthread_setspecific(PetscThreadsRankKey,&PetscThreadsRank[0]);
  /* Create threads */
  for(i=0; i<N; i++) {
    pVal_lockfree[i] = i;
    PetscThreadsRank[i+1] = i+1;
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
  PetscInt i;
  void* jstatus;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Signal all threads to finish */
  PetscThreadsRunKernel(PetscThreadsFinish_LockFree,NULL,PetscMaxThreads,PETSC_NULL);

  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);
  }

  ierr = pthread_setspecific(PetscThreadsRankKey,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscFree(pVal_lockfree);CHKERRQ(ierr);
  ierr = PetscFree(PetscThreadPoint);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.my_job_status);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.funcArr);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.pdata);CHKERRQ(ierr);
  ierr = PetscFree(busy_threads.list);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsWait_LockFree"
void* PetscThreadsWait_LockFree(void* arg) 
{
  PetscInt job_done,i;

  job_done = 0;
  /* Loop till all threads signal that they have done their job */
  while(!job_done) {
    for(i=0;i<busy_threads.nthreads;i++)
      job_done += job_lockfree.my_job_status[busy_threads.list[i]];
    if(job_done == busy_threads.nthreads) job_done = 1;
    else job_done = 0;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_LockFree"
PetscErrorCode PetscThreadsRunKernel_LockFree(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) 
{
  PetscInt i,j,k=0;

  busy_threads.nthreads = n-PetscMainThreadShareWork;
  PetscFunctionBegin;
  for(i=0;i<PetscMaxThreads;i++) {
    if(pFunc == PetscThreadsFinish_LockFree) {
      job_lockfree.funcArr[i+PetscMainThreadShareWork] = pFunc;
      job_lockfree.pdata[i+PetscMainThreadShareWork] = NULL;
      PetscAtomicCompareandSwap(&(job_lockfree.my_job_status[i]),1,0);
    } else {
      for(j=PetscMainThreadShareWork;j < n;j++) {
	if(cpu_affinity[j] == PetscThreadsCoreAffinities[i]) {
	  job_lockfree.funcArr[i+PetscMainThreadShareWork] = pFunc;
	  job_lockfree.pdata[i+PetscMainThreadShareWork] = data[j];
	  busy_threads.list[k++] = i;
	  /* signal thread i to start the job */
          PetscAtomicCompareandSwap(&(job_lockfree.my_job_status[i]),1,0);
	}
      }
    }
  }
  
  if(pFunc != PetscThreadsFinish_LockFree) {
    /* If the MainThreadShareWork flag is on then have the main thread also do the work */
    if(PetscMainThreadShareWork) {
      job_lockfree.funcArr[0] = pFunc;
      job_lockfree.pdata[0] = data[0];
      job_lockfree.funcArr[0](job_lockfree.pdata[0]);
    }
    /* Wait for all busy threads to finish their job */
    PetscThreadsWait(NULL);
  }

  PetscFunctionReturn(0);
}
