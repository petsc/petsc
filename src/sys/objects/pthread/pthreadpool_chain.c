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

PetscErrorCode ithreaderr_chain = 0;
int*           pVal_chain;

#define CACHE_LINE_SIZE 64
extern int* ThreadCoreAffinity;

typedef enum {JobInitiated,ThreadsWorking,JobCompleted} estat_chain;

typedef void* (*pfunc)(void*);

typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  pfunc* funcArr;
  void** pdata;
  PetscBool startJob;
  estat_chain eJobStat;
  PetscBool** arrThreadStarted;
  PetscBool** arrThreadReady;
} sjob_chain;
sjob_chain job_chain;

static pthread_cond_t main_cond_chain = PTHREAD_COND_INITIALIZER; 
static char* arrmutex;
static char* arrcond1;
static char* arrcond2;
static char* arrstart;
static char* arrready;

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

/*
 -----------------------------
    'Chain' Thread Functions 
 -----------------------------
*/
void* PetscThreadFunc_Chain(void* arg) {
  PetscErrorCode iterr;
  int ierr;
  int* pId = (int*)arg;
  int ThreadId = *pId;
  int SubWorker = ThreadId + 1;
  PetscBool PeeOn;

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);
#endif
  if(ThreadId==(PetscMaxThreads-1)) {
    PeeOn = PETSC_TRUE;
  }
  else {
    PeeOn = PETSC_FALSE;
  }
  if(PeeOn==PETSC_FALSE) {
    /* check your subordinate, wait for him to be ready */
    ierr = pthread_mutex_lock(job_chain.mutexarray[SubWorker]);
    while(*(job_chain.arrThreadReady[SubWorker])==PETSC_FALSE) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
      ierr = pthread_cond_wait(job_chain.cond1array[SubWorker],job_chain.mutexarray[SubWorker]);
    }
    ierr = pthread_mutex_unlock(job_chain.mutexarray[SubWorker]);
    /* your subordinate is now ready*/
  }
  ierr = pthread_mutex_lock(job_chain.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_chain.arrThreadReady[ThreadId]) = PETSC_TRUE;
  if(ThreadId==0) {
    job_chain.eJobStat = JobCompleted;
    /* signal main */
    ierr = pthread_cond_signal(&main_cond_chain);
  }
  else {
    /* tell your boss that you're ready to work */
    ierr = pthread_cond_signal(job_chain.cond1array[ThreadId]);
  }
  /*  the while loop needs to have an exit
     the 'main' thread can terminate all the threads by performing a broadcast
   and calling FuncFinish */
  while(PetscThreadGo) {
    /* need to check the condition to ensure we don't have to wait
       waiting when you don't have to causes problems
     also need to check the condition to ensure proper handling of spurious wakeups */
    while(*(job_chain.arrThreadReady[ThreadId])==PETSC_TRUE) {
      /*upon entry, automically releases the lock and blocks
       upon return, has the lock */
        ierr = pthread_cond_wait(job_chain.cond2array[ThreadId],job_chain.mutexarray[ThreadId]);
	*(job_chain.arrThreadStarted[ThreadId]) = PETSC_TRUE;
	*(job_chain.arrThreadReady[ThreadId])   = PETSC_FALSE;
    }
    if(ThreadId==0) {
      job_chain.startJob = PETSC_FALSE;
      job_chain.eJobStat = ThreadsWorking;
    }
    ierr = pthread_mutex_unlock(job_chain.mutexarray[ThreadId]);
    if(PeeOn==PETSC_FALSE) {
      /* tell your subworker it's time to get to work */
      ierr = pthread_cond_signal(job_chain.cond2array[SubWorker]);
    }
    /* do your job */
    if(job_chain.funcArr[ThreadId+PetscMainThreadShareWork]) {
      iterr = (PetscErrorCode)(long int)job_chain.funcArr[ThreadId+PetscMainThreadShareWork](job_chain.pdata[ThreadId+PetscMainThreadShareWork]);
    }
    if(iterr!=0) {
      ithreaderr_chain = 1;
    }
    if(PetscThreadGo) {
      /* reset job, get ready for more */
      if(PeeOn==PETSC_FALSE) {
        /* check your subordinate, wait for him to be ready
         how do you know for a fact that your subordinate has actually started? */
        ierr = pthread_mutex_lock(job_chain.mutexarray[SubWorker]);
        while(*(job_chain.arrThreadReady[SubWorker])==PETSC_FALSE||*(job_chain.arrThreadStarted[SubWorker])==PETSC_FALSE) {
          /* upon entry, automically releases the lock and blocks
           upon return, has the lock */
          ierr = pthread_cond_wait(job_chain.cond1array[SubWorker],job_chain.mutexarray[SubWorker]);
        }
        ierr = pthread_mutex_unlock(job_chain.mutexarray[SubWorker]);
        /* your subordinate is now ready */
      }
      ierr = pthread_mutex_lock(job_chain.mutexarray[ThreadId]);
      *(job_chain.arrThreadReady[ThreadId]) = PETSC_TRUE;
      if(ThreadId==0) {
	job_chain.eJobStat = JobCompleted; /* foreman: last thread to complete, guaranteed! */
        /* root thread (foreman) signals 'main' */
        ierr = pthread_cond_signal(&main_cond_chain);
      }
      else {
        /* signal your boss before you go to sleep */
        ierr = pthread_cond_signal(job_chain.cond1array[ThreadId]);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_Chain"
PetscErrorCode PetscThreadInitialize_Chain(PetscInt N) 
{
  PetscErrorCode ierr;
  PetscInt i,status;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_MEMALIGN)
  size_t Val1 = (size_t)CACHE_LINE_SIZE;
  size_t Val2 = (size_t)PetscMaxThreads*CACHE_LINE_SIZE;
  arrmutex = (char*)memalign(Val1,Val2);
  arrcond1 = (char*)memalign(Val1,Val2);
  arrcond2 = (char*)memalign(Val1,Val2);
  arrstart = (char*)memalign(Val1,Val2);
  arrready = (char*)memalign(Val1,Val2);
#else
  size_t Val2 = (size_t)PetscMaxThreads*CACHE_LINE_SIZE;
  arrmutex = (char*)malloc(Val2);
  arrcond1 = (char*)malloc(Val2);
  arrcond2 = (char*)malloc(Val2);
  arrstart = (char*)malloc(Val2);
  arrready = (char*)malloc(Val2);
#endif
  
  job_chain.mutexarray       = (pthread_mutex_t**)malloc(PetscMaxThreads*sizeof(pthread_mutex_t*));
  job_chain.cond1array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_chain.cond2array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_chain.arrThreadStarted = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
  job_chain.arrThreadReady   = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
  /* initialize job structure */
  for(i=0; i<PetscMaxThreads; i++) {
    job_chain.mutexarray[i]        = (pthread_mutex_t*)(arrmutex+CACHE_LINE_SIZE*i);
    job_chain.cond1array[i]        = (pthread_cond_t*)(arrcond1+CACHE_LINE_SIZE*i);
    job_chain.cond2array[i]        = (pthread_cond_t*)(arrcond2+CACHE_LINE_SIZE*i);
    job_chain.arrThreadStarted[i]  = (PetscBool*)(arrstart+CACHE_LINE_SIZE*i);
    job_chain.arrThreadReady[i]    = (PetscBool*)(arrready+CACHE_LINE_SIZE*i);
  }
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_mutex_init(job_chain.mutexarray[i],NULL);
    ierr = pthread_cond_init(job_chain.cond1array[i],NULL);
    ierr = pthread_cond_init(job_chain.cond2array[i],NULL);
    *(job_chain.arrThreadStarted[i])  = PETSC_FALSE;
    *(job_chain.arrThreadReady[i])    = PETSC_FALSE;
  }
  job_chain.funcArr = (pfunc*)malloc((N+PetscMainThreadShareWork)*sizeof(pfunc));
  job_chain.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  job_chain.startJob = PETSC_FALSE;
  job_chain.eJobStat = JobInitiated;
  pVal_chain = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  /* create threads */
  for(i=0; i<N; i++) {
    pVal_chain[i] = i;
    job_chain.funcArr[i+PetscMainThreadShareWork] = NULL;
    job_chain.pdata[i+PetscMainThreadShareWork] = NULL;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_chain[i]);
    /* should check error */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_Chain"
PetscErrorCode PetscThreadFinalize_Chain() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads,PETSC_NULL);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
    /* should check error */
  }
  free(PetscThreadPoint);
  free(arrmutex);
  free(arrcond1);
  free(arrcond2);
  free(arrstart);
  free(arrready);
  free(job_chain.pdata);
  free(job_chain.funcArr);
  free(pVal_chain);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Chain"
void* MainWait_Chain(void* arg) {
  int ierr;
  ierr = pthread_mutex_lock(job_chain.mutexarray[0]);
  while(job_chain.eJobStat<JobCompleted||job_chain.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond_chain,job_chain.mutexarray[0]);
  }
  ierr = pthread_mutex_unlock(job_chain.mutexarray[0]);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_Chain"
PetscErrorCode MainJob_Chain(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) {
  int i,j,issetaffinity,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait(NULL);
  job_chain.startJob = PETSC_TRUE;
  for(i=0; i<PetscMaxThreads; i++) {
    if(pFunc == FuncFinish) {
      job_chain.funcArr[i+PetscMainThreadShareWork] = pFunc;
      job_chain.pdata[i+PetscMainThreadShareWork] = NULL;
    } else {
      issetaffinity=0;
      for(j=PetscMainThreadShareWork; j < n; j++) {
	if(cpu_affinity[j] == ThreadCoreAffinity[i]) {
	  job_chain.funcArr[i+PetscMainThreadShareWork] = pFunc;
	  job_chain.pdata[i+PetscMainThreadShareWork] = data[j];
	  issetaffinity=1;
	}
      }
      if(!issetaffinity) {
	job_chain.funcArr[i+PetscMainThreadShareWork] = NULL;
	job_chain.pdata[i+PetscMainThreadShareWork] = NULL;
      }
    }
    *(job_chain.arrThreadStarted[i]) = PETSC_FALSE;
  }
  job_chain.eJobStat = JobInitiated;
  ierr = pthread_cond_signal(job_chain.cond2array[0]);
  if(pFunc!=FuncFinish) {
    if(PetscMainThreadShareWork) {
      job_chain.funcArr[0] = pFunc;
      job_chain.pdata[0] = data[0];
      ijoberr = (PetscErrorCode)(long int)job_chain.funcArr[0](job_chain.pdata[0]);
    }
    MainWait(NULL); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr_chain) {
    ijoberr = ithreaderr_chain;
  }
  return ijoberr;
}
