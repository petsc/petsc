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

PetscErrorCode ithreaderr_true = 0;
int*           pVal_true;
#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
static pthread_barrier_t* BarrPoint;   /* used by 'true' thread pool */
#endif

#define CACHE_LINE_SIZE 64
extern int* ThreadCoreAffinity;

/* true thread pool data structure */
#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  void* (*pfunc)(void*);
  void** pdata;
  pthread_barrier_t* pbarr;
  int iNumJobThreads;
  int iNumReadyThreads;
  PetscBool startJob;
} sjob_true;
sjob_true job_true = {PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER,NULL,NULL,NULL,0,0,PETSC_FALSE};
#endif

static pthread_cond_t main_cond_true = PTHREAD_COND_INITIALIZER; 

/* external Functions */
extern void*          (*PetscThreadFunc)(void*);
extern PetscErrorCode (*PetscThreadInitialize)(PetscInt);
extern PetscErrorCode (*PetscThreadFinalize)(void);
extern void           (*MainWait)(void);
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt);

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
  int ierr,iVal;
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
    iVal = PetscMaxThreads-job_true.iNumReadyThreads-1;
    pthread_mutex_unlock(&job_true.mutex);
    if(job_true.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_true.pfunc(job_true.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_true.pfunc(job_true.pdata[iVal]);
    }
    if(iterr!=0) {
      ithreaderr_true = 1;
    }

    /* the barrier is necessary BECAUSE: look at job_true.iNumReadyThreads
      what happens if a thread finishes before they all start? BAD!
     what happens if a thread finishes before any else start? BAD! */
    pthread_barrier_wait(job_true.pbarr); /* ensures all threads are finished */
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
  BarrPoint = (pthread_barrier_t*)malloc((N+1)*sizeof(pthread_barrier_t)); /* BarrPoint[0] makes no sense, don't use it! */
  job_true.pdata = (void**)malloc(N*sizeof(void*));
  for(i=0; i<N; i++) {
    pVal_true[i] = i;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_true[i]);
    /* error check to ensure proper thread creation */
    status = pthread_barrier_init(&BarrPoint[i+1],NULL,i+1);
    /* should check error */
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_True"
PetscErrorCode PetscThreadFinalize_True() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
  }
  free(BarrPoint);
  free(PetscThreadPoint);
  free(pVal_true);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_True"
void MainWait_True() {
  int ierr;
  ierr = pthread_mutex_lock(&job_true.mutex);
  while(job_true.iNumReadyThreads<PetscMaxThreads||job_true.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond_true,&job_true.mutex);
  }
  ierr = pthread_mutex_unlock(&job_true.mutex);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_True"
PetscErrorCode MainJob_True(void* (*pFunc)(void*),void** data,PetscInt n) {
  int ierr;
  PetscErrorCode ijoberr = 0;

  MainWait();
  job_true.pfunc = pFunc;
  job_true.pdata = data;
  job_true.pbarr = &BarrPoint[n];
  job_true.iNumJobThreads = n;
  job_true.startJob = PETSC_TRUE;
  ierr = pthread_cond_broadcast(&job_true.cond);
  if(pFunc!=FuncFinish) {
    MainWait(); /* why wait after? guarantees that job gets done */
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
