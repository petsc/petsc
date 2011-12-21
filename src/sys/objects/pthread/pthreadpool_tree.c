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

static PetscErrorCode ithreaderr_tree = 0;
static int*         pVal_tree;

#define CACHE_LINE_SIZE 64
extern int* ThreadCoreAffinity;

typedef enum {JobInitiated,ThreadsWorking,JobCompleted} estat_tree;

/* Tree thread pool data structure */
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  void* (*pfunc)(void*);
  void** pdata;
  PetscBool startJob;
  estat_tree eJobStat;
  PetscBool** arrThreadStarted;
  PetscBool** arrThreadReady;
} sjob_tree;
sjob_tree job_tree;

static pthread_cond_t main_cond_tree = PTHREAD_COND_INITIALIZER; 
static char* arrmutex;
static char* arrcond1;
static char* arrcond2;
static char* arrstart;
static char* arrready;

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

/*
  'Tree' Thread Pool Functions 
*/
void* PetscThreadFunc_Tree(void* arg) {
  PetscErrorCode iterr;
  int ierr;
  int* pId = (int*)arg;
  int ThreadId = *pId,Mary = 2,i,SubWorker;
  PetscBool PeeOn;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);
#endif

  if((Mary*ThreadId+1)>(PetscMaxThreads-1)) {
    PeeOn = PETSC_TRUE;
  }
  else {
    PeeOn = PETSC_FALSE;
  }
  if(PeeOn==PETSC_FALSE) {
    /* check your subordinates, wait for them to be ready */
    for(i=1;i<=Mary;i++) {
      SubWorker = Mary*ThreadId+i;
      if(SubWorker<PetscMaxThreads) {
        ierr = pthread_mutex_lock(job_tree.mutexarray[SubWorker]);
        while(*(job_tree.arrThreadReady[SubWorker])==PETSC_FALSE) {
          /* upon entry, automically releases the lock and blocks
           upon return, has the lock */
          ierr = pthread_cond_wait(job_tree.cond1array[SubWorker],job_tree.mutexarray[SubWorker]);
        }
        ierr = pthread_mutex_unlock(job_tree.mutexarray[SubWorker]);
      }
    }
    /* your subordinates are now ready */
  }
  ierr = pthread_mutex_lock(job_tree.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_tree.arrThreadReady[ThreadId]) = PETSC_TRUE;
  if(ThreadId==0) {
    job_tree.eJobStat = JobCompleted;
    /* ignal main */
    ierr = pthread_cond_signal(&main_cond_tree);
  }
  else {
    /* tell your boss that you're ready to work */
    ierr = pthread_cond_signal(job_tree.cond1array[ThreadId]);
  }
  /* the while loop needs to have an exit
  the 'main' thread can terminate all the threads by performing a broadcast
   and calling FuncFinish */
  while(PetscThreadGo) {
    /*need to check the condition to ensure we don't have to wait
      waiting when you don't have to causes problems
     also need to check the condition to ensure proper handling of spurious wakeups */
    while(*(job_tree.arrThreadReady[ThreadId])==PETSC_TRUE) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
        ierr = pthread_cond_wait(job_tree.cond2array[ThreadId],job_tree.mutexarray[ThreadId]);
	*(job_tree.arrThreadStarted[ThreadId]) = PETSC_TRUE;
	*(job_tree.arrThreadReady[ThreadId])   = PETSC_FALSE;
    }
    if(ThreadId==0) {
      job_tree.startJob = PETSC_FALSE;
      job_tree.eJobStat = ThreadsWorking;
    }
    ierr = pthread_mutex_unlock(job_tree.mutexarray[ThreadId]);
    if(PeeOn==PETSC_FALSE) {
      /* tell your subordinates it's time to get to work */
      for(i=1; i<=Mary; i++) {
	SubWorker = Mary*ThreadId+i;
        if(SubWorker<PetscMaxThreads) {
          ierr = pthread_cond_signal(job_tree.cond2array[SubWorker]);
        }
      }
    }
    /* do your job */
    if(job_tree.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_tree.pfunc(job_tree.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_tree.pfunc(job_tree.pdata[ThreadId+PetscMainThreadShareWork]);
    }
    if(iterr!=0) {
      ithreaderr_tree = 1;
    }
    if(PetscThreadGo) {
      /* reset job, get ready for more */
      if(PeeOn==PETSC_FALSE) {
        /* check your subordinates, waiting for them to be ready
         how do you know for a fact that a given subordinate has actually started? */
	for(i=1;i<=Mary;i++) {
	  SubWorker = Mary*ThreadId+i;
          if(SubWorker<PetscMaxThreads) {
            ierr = pthread_mutex_lock(job_tree.mutexarray[SubWorker]);
            while(*(job_tree.arrThreadReady[SubWorker])==PETSC_FALSE||*(job_tree.arrThreadStarted[SubWorker])==PETSC_FALSE) {
              /* upon entry, automically releases the lock and blocks
               upon return, has the lock */
              ierr = pthread_cond_wait(job_tree.cond1array[SubWorker],job_tree.mutexarray[SubWorker]);
            }
            ierr = pthread_mutex_unlock(job_tree.mutexarray[SubWorker]);
          }
	}
        /* your subordinates are now ready */
      }
      ierr = pthread_mutex_lock(job_tree.mutexarray[ThreadId]);
      *(job_tree.arrThreadReady[ThreadId]) = PETSC_TRUE;
      if(ThreadId==0) {
	job_tree.eJobStat = JobCompleted; /* oot thread: last thread to complete, guaranteed! */
        /* root thread signals 'main' */
        ierr = pthread_cond_signal(&main_cond_tree);
      }
      else {
        /* signal your boss before you go to sleep */
        ierr = pthread_cond_signal(job_tree.cond1array[ThreadId]);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_Tree"
PetscErrorCode PetscThreadInitialize_Tree(PetscInt N) 
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       status;

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
  job_tree.mutexarray       = (pthread_mutex_t**)malloc(PetscMaxThreads*sizeof(pthread_mutex_t*));
  job_tree.cond1array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_tree.cond2array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_tree.arrThreadStarted = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
  job_tree.arrThreadReady   = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
  /* initialize job structure */
  for(i=0; i<PetscMaxThreads; i++) {
    job_tree.mutexarray[i]        = (pthread_mutex_t*)(arrmutex+CACHE_LINE_SIZE*i);
    job_tree.cond1array[i]        = (pthread_cond_t*)(arrcond1+CACHE_LINE_SIZE*i);
    job_tree.cond2array[i]        = (pthread_cond_t*)(arrcond2+CACHE_LINE_SIZE*i);
    job_tree.arrThreadStarted[i]  = (PetscBool*)(arrstart+CACHE_LINE_SIZE*i);
    job_tree.arrThreadReady[i]    = (PetscBool*)(arrready+CACHE_LINE_SIZE*i);
  }
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_mutex_init(job_tree.mutexarray[i],NULL);
    ierr = pthread_cond_init(job_tree.cond1array[i],NULL);
    ierr = pthread_cond_init(job_tree.cond2array[i],NULL);
    *(job_tree.arrThreadStarted[i])  = PETSC_FALSE;
    *(job_tree.arrThreadReady[i])    = PETSC_FALSE;
  }
  job_tree.pfunc = NULL;
  job_tree.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  job_tree.startJob = PETSC_FALSE;
  job_tree.eJobStat = JobInitiated;
  pVal_tree = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  /* create threads */
  for(i=0; i<N; i++) {
    pVal_tree[i] = i;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_tree[i]);
    /* should check status */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_Tree"
PetscErrorCode PetscThreadFinalize_Tree() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
    /* do error checking*/
  }
  free(PetscThreadPoint);
  free(arrmutex);
  free(arrcond1);
  free(arrcond2);
  free(arrstart);
  free(arrready);
  free(job_tree.pdata);
  free(pVal_tree);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Tree"
void MainWait_Tree() {
  int ierr;
  ierr = pthread_mutex_lock(job_tree.mutexarray[0]);
  while(job_tree.eJobStat<JobCompleted||job_tree.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond_tree,job_tree.mutexarray[0]);
  }
  ierr = pthread_mutex_unlock(job_tree.mutexarray[0]);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_Tree"
PetscErrorCode MainJob_Tree(void* (*pFunc)(void*),void** data,PetscInt n) {
  int i,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait();
  job_tree.pfunc = pFunc;
  job_tree.pdata = data;
  job_tree.startJob = PETSC_TRUE;
  for(i=0; i<PetscMaxThreads; i++) {
    *(job_tree.arrThreadStarted[i]) = PETSC_FALSE;
  }
  job_tree.eJobStat = JobInitiated;
  ierr = pthread_cond_signal(job_tree.cond2array[0]);
  if(pFunc!=FuncFinish) {
    if(PetscMainThreadShareWork) {
      ijoberr = (PetscErrorCode)(long int)job_tree.pfunc(job_tree.pdata[0]);
    }
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr_tree) {
    ijoberr = ithreaderr_tree;
  }
  return ijoberr;
}
