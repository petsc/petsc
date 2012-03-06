#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

PetscInt *pVal_none;

typedef void* (*pfunc)(void*);
typedef struct {
  pfunc* funcArr;
  int   nthreads;
  void** pdata;
  pthread_t* ThreadId;
}sjob_none;

sjob_none job_none;

/* 
   -----------------------------
     'NO' THREAD POOL FUNCTION 
   -----------------------------
*/
void* PetscThreadFunc_None(void* arg)
{
  int iVal;

  iVal = *(int*)arg;
  pthread_setspecific(rankkey,&threadranks[iVal+1]);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  DoCoreAffinity();
#endif

  job_none.funcArr[iVal+PetscMainThreadShareWork](job_none.pdata[iVal+PetscMainThreadShareWork]);

  pthread_setspecific(rankkey,NULL);
  return NULL;
}
  
void* PetscThreadsWait_None(void* arg)
{
  int            nthreads;
  PetscInt       i;
  void*          joinstatus;

  nthreads = *(PetscInt*)arg;
  for (i=0; i<nthreads; i++) {
    pthread_join(PetscThreadPoint[i], &joinstatus);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_None"
PetscErrorCode PetscThreadsRunKernel_None(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) 
{
  PetscInt i;
  PetscInt Nnew_threads=n-PetscMainThreadShareWork;

  PetscFunctionBegin;
  pVal_none = (int*)malloc((n-PetscMainThreadShareWork)*sizeof(int));
  PetscThreadPoint = (pthread_t*)malloc((n-PetscMainThreadShareWork)*sizeof(pthread_t));
  job_none.funcArr = (pfunc*)malloc(n*sizeof(pfunc));
  job_none.pdata   = (void**)malloc(n*sizeof(void*));

  threadranks[0] = 0;
  pthread_setspecific(rankkey,&threadranks[0]);
  for(i=0;i< Nnew_threads;i++) {
    pVal_none[i] = i;
    threadranks[i+1] = i+1;
    ThreadCoreAffinity[i] = cpu_affinity[i+PetscMainThreadShareWork];
    job_none.funcArr[i+PetscMainThreadShareWork] = pFunc;
    job_none.pdata[i+PetscMainThreadShareWork]  = data[i+PetscMainThreadShareWork];
    pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc_None,&pVal_none[i]);
  }
  if(PetscMainThreadShareWork) pFunc(data[0]);

  PetscThreadsWait(&Nnew_threads);

  free(PetscThreadPoint);
  free(job_none.funcArr);
  free(job_none.pdata);
  free(pVal_none);

  PetscFunctionReturn(0);
}
