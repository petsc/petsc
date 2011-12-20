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

static PetscErrorCode ithreaderr = 0;
static int*         pVal_main;

#define CACHE_LINE_SIZE 64  /* used by 'chain', 'main','tree' thread pools */
extern int* ThreadCoreAffinity;

/* main thread pool data structure */
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  void* (*pfunc)(void*);
  void** pdata;
  PetscBool** arrThreadReady;
} sjob_main;
sjob_main job_main;

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
   ----------------------------
   'Main' Thread Pool Functions
   ---------------------------- 
*/
void* PetscThreadFunc_Main(void* arg) {
  PetscErrorCode iterr;
  int ierr;
  int* pId = (int*)arg;
  int ThreadId = *pId;

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);
#endif

  ierr = pthread_mutex_lock(job_main.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_main.arrThreadReady[ThreadId]) = PETSC_TRUE;
  /* tell the BOSS that you're ready to work before you go to sleep */
  ierr = pthread_cond_signal(job_main.cond1array[ThreadId]);

  /* the while loop needs to have an exit
     the 'main' thread can terminate all the threads by performing a broadcast
     and calling FuncFinish */
  while(PetscThreadGo) {
    /* need to check the condition to ensure we don't have to wait
       waiting when you don't have to causes problems
     also need to check the condition to ensure proper handling of spurious wakeups */
    while(*(job_main.arrThreadReady[ThreadId])==PETSC_TRUE) {
      /* upon entry, atomically releases the lock and blocks
       upon return, has the lock */
        ierr = pthread_cond_wait(job_main.cond2array[ThreadId],job_main.mutexarray[ThreadId]);
	/* (job_main.arrThreadReady[ThreadId])   = PETSC_FALSE; */
    }
    ierr = pthread_mutex_unlock(job_main.mutexarray[ThreadId]);
    if(job_main.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_main.pfunc(job_main.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_main.pfunc(job_main.pdata[ThreadId+PetscMainThreadShareWork]);
    }
    if(iterr!=0) {
      ithreaderr = 1;
    }
    if(PetscThreadGo) {
      /* reset job, get ready for more */
      ierr = pthread_mutex_lock(job_main.mutexarray[ThreadId]);
      *(job_main.arrThreadReady[ThreadId]) = PETSC_TRUE;
      /* tell the BOSS that you're ready to work before you go to sleep */
      ierr = pthread_cond_signal(job_main.cond1array[ThreadId]);
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_Main"
PetscErrorCode PetscThreadInitialize_Main(PetscInt N) 
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
  
  job_main.mutexarray       = (pthread_mutex_t**)malloc(PetscMaxThreads*sizeof(pthread_mutex_t*));
  job_main.cond1array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_main.cond2array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
  job_main.arrThreadReady   = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
  /* initialize job structure */
  for(i=0; i<PetscMaxThreads; i++) {
    job_main.mutexarray[i]        = (pthread_mutex_t*)(arrmutex+CACHE_LINE_SIZE*i);
    job_main.cond1array[i]        = (pthread_cond_t*)(arrcond1+CACHE_LINE_SIZE*i);
    job_main.cond2array[i]        = (pthread_cond_t*)(arrcond2+CACHE_LINE_SIZE*i);
    job_main.arrThreadReady[i]    = (PetscBool*)(arrready+CACHE_LINE_SIZE*i);
  }
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_mutex_init(job_main.mutexarray[i],NULL);
    ierr = pthread_cond_init(job_main.cond1array[i],NULL);
    ierr = pthread_cond_init(job_main.cond2array[i],NULL);
    *(job_main.arrThreadReady[i])    = PETSC_FALSE;
  }
  job_main.pfunc = NULL;
  job_main.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  pVal_main = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  /* create threads */
  for(i=0; i<N; i++) {
    pVal_main[i] = i;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_main[i]);
    /* error check */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_Main"
PetscErrorCode PetscThreadFinalize_Main() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);
  }
  free(PetscThreadPoint);
  free(arrmutex);
  free(arrcond1);
  free(arrcond2);
  free(arrstart);
  free(arrready);
  free(job_main.pdata);
  free(pVal_main);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Main"
void MainWait_Main() {
  int i,ierr;
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_mutex_lock(job_main.mutexarray[i]);
    while(*(job_main.arrThreadReady[i])==PETSC_FALSE) {
      ierr = pthread_cond_wait(job_main.cond1array[i],job_main.mutexarray[i]);
    }
    ierr = pthread_mutex_unlock(job_main.mutexarray[i]);
  }
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_Main"
PetscErrorCode MainJob_Main(void* (*pFunc)(void*),void** data,PetscInt n) {
  int i,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait(); /* you know everyone is waiting to be signalled! */
  job_main.pfunc = pFunc;
  job_main.pdata = data;
  for(i=0; i<PetscMaxThreads; i++) {
    *(job_main.arrThreadReady[i]) = PETSC_FALSE; /* why do this?  suppose you get into MainWait first */
  }
  /* tell the threads to go to work */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_cond_signal(job_main.cond2array[i]);
  }
  if(pFunc!=FuncFinish) {
    if(PetscMainThreadShareWork) {
      ijoberr = (PetscErrorCode)(long int)job_main.pfunc(job_main.pdata[0]);
    }
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}
