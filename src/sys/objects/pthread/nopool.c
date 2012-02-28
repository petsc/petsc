#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

typedef void* (*pfunc)(void*);
typedef struct {
  pfunc kernelfunc;
  int   nthreads;
  void** data;
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
  sjob_none         *job = (sjob_none*)arg;
  PetscInt          i;
  PetscInt          nthreads= (int)job->nthreads;
  void**            data = (void**)job->data;
  pthread_t*        ThreadId = (pthread_t*)job->ThreadId;
  pfunc             funcp = (pfunc)job->kernelfunc;

  for(i=0; i<nthreads; i++) {
    pthread_create(&ThreadId[i],NULL,funcp,data[i]);
  }
  return(0);
}

void* PetscThreadsWait_None(void* arg)
{
  sjob_none      *job=(sjob_none*)arg;
  int            nthreads = job->nthreads;
  pthread_t*     ThreadId = (pthread_t*)job->ThreadId;
  PetscInt       i;
  void*          joinstatus;
  for (i=0; i<nthreads; i++) {
    pthread_join(ThreadId[i], &joinstatus);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_None"
PetscErrorCode PetscThreadsRunKernel_None(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) 
{
  PetscInt i;

  PetscFunctionBegin;
  pthread_t* apThread = (pthread_t*)malloc(n*sizeof(pthread_t));
  PetscThreadPoint = apThread; /* point to same place */
  job_none.nthreads = n;
  job_none.kernelfunc = pFunc;
  job_none.data = data;
  job_none.ThreadId = apThread;
  for(i=0;i<n;i++) ThreadCoreAffinity[i] = cpu_affinity[i];
  PetscThreadFunc(&job_none);
  PetscThreadsWait(&job_none); /* ensures that all threads are finished with the job */
  free(apThread);

  PetscFunctionReturn(0);
}
