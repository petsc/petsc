/* The code is active only when the flag PETSC_USE_PTHREAD is set */


#include <petscsys.h>        /*I  "petscsys.h"   I*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#if defined(PETSC_HAVE_SCHED_H)
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif
#if defined(PETSC_HAVE_PTHREAD_H)
#include <pthread.h>
#endif

#if defined(PETSC_HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#if defined(PETSC_HAVE_VALGRIND)
#include <valgrind/valgrind.h>
#endif

extern PetscBool    PetscThreadGo;
extern PetscMPIInt  PetscMaxThreads;
extern pthread_t*   PetscThreadPoint;
extern PetscInt     PetscMainThreadShareWork;

PetscErrorCode ithreaderr_true = 0;
int*           pVal_true;

#define CACHE_LINE_SIZE 64
extern int* ThreadCoreAffinity;

typedef void* (*pfunc)(void*);

/* true thread pool data structure */
#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
pthread_barrier_t pbarr;
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pfunc *funcArr;
  void** pdata;
  int iNumJobThreads;
  int iNumReadyThreads;
  PetscBool startJob;
} sjob_true;

sjob_true job_true = {PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER,NULL,NULL,0,0,PETSC_FALSE};
#endif

static pthread_cond_t main_cond_true = PTHREAD_COND_INITIALIZER; 

/* external Functions */
extern void*          (*PetscThreadFunc)(void*);
extern PetscErrorCode (*PetscThreadInitialize)(PetscInt);
extern PetscErrorCode (*PetscThreadFinalize)(void);
extern void*           (*MainWait)(void*);
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
extern void PetscPthreadSetAffinity(PetscInt);
#endif

extern void* FuncFinish(void*);

#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
/* 
  ----------------------------
     'True' Thread Functions 
  ----------------------------
*/
void* PetscThreadFunc_True(void* arg) {
  int ierr;
  PetscErrorCode iterr;

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  int* pId      = (int*)arg;
  int  ThreadId = *pId; 
  PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);
#endif
  ierr = pthread_mutex_lock(&job_true.mutex);
  job_true.iNumReadyThreads++;
  if(job_true.iNumReadyThreads==PetscMaxThreads) {
    ierr = pthread_cond_signal(&main_cond_true);
  }
  /*the while loop needs to have an exit
    the 'main' thread can terminate all the threads by performing a broadcast
   and calling FuncFinish */
  while(PetscThreadGo) {
    /*need to check the condition to ensure we don't have to wait
      waiting when you don't have to causes problems
     also need to wait if another thread sneaks in and messes with the predicate */
    while(job_true.startJob==PETSC_FALSE&&job_true.iNumJobThreads==0) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
      ierr = pthread_cond_wait(&job_true.cond,&job_true.mutex);
    }
    job_true.startJob = PETSC_FALSE;
    job_true.iNumJobThreads--;
    job_true.iNumReadyThreads--;
    pthread_mutex_unlock(&job_true.mutex);

    if(job_true.funcArr[ThreadId+PetscMainThreadShareWork]) {
      iterr = (PetscErrorCode)(long int)job_true.funcArr[ThreadId+PetscMainThreadShareWork](job_true.pdata[ThreadId+PetscMainThreadShareWork]);
    }
    if(iterr!=0) {
      ithreaderr_true = 1;
    }

    /* the barrier is necessary BECAUSE: look at job_true.iNumReadyThreads
      what happens if a thread finishes before they all start? BAD!
     what happens if a thread finishes before any else start? BAD! */
    pthread_barrier_wait(&pbarr); /* ensures all threads are finished */
    /* reset job */
    if(PetscThreadGo) {
      pthread_mutex_lock(&job_true.mutex);
      job_true.iNumReadyThreads++;
      if(job_true.iNumReadyThreads==PetscMaxThreads) {
	/* signal the 'main' thread that the job is done! (only done once) */
	ierr = pthread_cond_signal(&main_cond_true);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_True"
PetscErrorCode PetscThreadInitialize_True(PetscInt N)
{
  PetscInt i,status;

  PetscFunctionBegin;
  pVal_true = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  /* Initialize the barrier */
  status = pthread_barrier_init(&pbarr,NULL,PetscMaxThreads);
  job_true.funcArr = (pfunc*)malloc((N+PetscMainThreadShareWork)*sizeof(pfunc));
  job_true.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  for(i=0; i<N; i++) {
    pVal_true[i] = i;
    job_true.funcArr[i+PetscMainThreadShareWork] = NULL;
    job_true.pdata[i+PetscMainThreadShareWork] = NULL;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_true[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_True"
PetscErrorCode PetscThreadFinalize_True() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads,PETSC_NULL);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
  }
  free(PetscThreadPoint);
  free(job_true.funcArr);
  free(job_true.pdata);
  free(pVal_true);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_True"
void* MainWait_True(void* arg) {
  int ierr;
  ierr = pthread_mutex_lock(&job_true.mutex);
  while(job_true.iNumReadyThreads<PetscMaxThreads||job_true.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond_true,&job_true.mutex);
  }
  ierr = pthread_mutex_unlock(&job_true.mutex);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_True"
PetscErrorCode MainJob_True(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) {
  int i,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait(NULL);
  for(i=0; i<PetscMaxThreads; i++) {
    if(pFunc == FuncFinish) {
      job_true.funcArr[i+PetscMainThreadShareWork] = pFunc;
      job_true.pdata[i+PetscMainThreadShareWork] = NULL;
    } else {
      /* Currently this model assumes that the first n threads will be only doing the useful work while
	 the remaining threads will be just spinning.
	 Need to modify this model when threads with specific affinities, e.g., n threads pinned to only
	 one socket,or n threads spread across different sockets, are requested.
      */
      if (i < n-PetscMainThreadShareWork) {
	job_true.funcArr[i+PetscMainThreadShareWork] = pFunc;
	job_true.pdata[i+PetscMainThreadShareWork] = data[i+PetscMainThreadShareWork];
      }
      else {
	job_true.funcArr[i+PetscMainThreadShareWork] = NULL;
	job_true.pdata[i+PetscMainThreadShareWork] = NULL;
      }
    }
  }

  job_true.iNumJobThreads = PetscMaxThreads;;
  job_true.startJob = PETSC_TRUE;
  /* Tell the threads to go to work */
  ierr = pthread_cond_broadcast(&job_true.cond);
  if(pFunc!=FuncFinish) {
    if(PetscMainThreadShareWork) {
      job_true.funcArr[0] = pFunc;
      job_true.pdata[0] = data[0];
      ijoberr = (PetscErrorCode)(long int)job_true.funcArr[0](job_true.pdata[0]);
    }
    MainWait(NULL); /* why wait after? guarantees that job gets done */
  }

  if(ithreaderr_true) {
    ijoberr = ithreaderr_true;
  }
  return ijoberr;
}
#else
int PetscPthread_dummy()
{
  return 0;
}
#endif
