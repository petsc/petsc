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
#if defined(PETSC_HAVE_SYS_SYSCTL_H)
#include <sys/sysctl.h>
#endif

PetscBool    PetscCheckCoreAffinity    = PETSC_FALSE;
PetscBool    PetscThreadGo         = PETSC_TRUE;
PetscMPIInt  PetscMaxThreads = -1; /* Later set when PetscSetMaxPThreads is called */
pthread_t*   PetscThreadPoint;
int*         ThreadCoreAffinity;
PetscInt     PetscMainThreadShareWork = 1; /* Flag to indicate whether the main thread shares work along with the worker threads, 1 by default, can be switched off using option -mainthread_no_share_work */
PetscInt     MainThreadCoreAffinity=0;
PetscInt     N_CORES;

typedef enum {THREADSYNC_NOPOOL,THREADSYNC_MAINPOOL,THREADSYNC_TRUEPOOL,THREADSYNC_CHAINPOOL,THREADSYNC_TREEPOOL,THREADSYNC_LOCKFREE} ThreadSynchronizationType;
static const char *ThreadSynchronizationTypes[] = {"NOPOOL","MAINPOOL","TRUEPOOL","CHAINPOOL","TREEPOOL","LOCKFREE","ThreadSynchronizationType","THREADSYNC_",0};

/* Function Pointers */
void*          (*PetscThreadFunc)(void*) = NULL;
PetscErrorCode (*PetscThreadInitialize)(PetscInt) = NULL;
PetscErrorCode (*PetscThreadFinalize)(void) = NULL;
void*          (*MainWait)(void*) = NULL;
						  PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt,PetscInt*) = NULL;

/* Tree Thread Pool Functions */
extern void*          PetscThreadFunc_Tree(void*);
extern PetscErrorCode PetscThreadInitialize_Tree(PetscInt);
extern PetscErrorCode PetscThreadFinalize_Tree(void);
extern void*          MainWait_Tree(void*);
						  extern PetscErrorCode MainJob_Tree(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Main Thread Pool Functions */
extern void*          PetscThreadFunc_Main(void*);
extern PetscErrorCode PetscThreadInitialize_Main(PetscInt);
extern PetscErrorCode PetscThreadFinalize_Main(void);
extern void*          MainWait_Main(void*);
						  extern PetscErrorCode MainJob_Main(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Chain Thread Pool Functions */
extern void*          PetscThreadFunc_Chain(void*);
extern PetscErrorCode PetscThreadInitialize_Chain(PetscInt);
extern PetscErrorCode PetscThreadFinalize_Chain(void);
extern void*          MainWait_Chain(void*);
						  extern PetscErrorCode MainJob_Chain(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* True Thread Pool Functions */
extern void*          PetscThreadFunc_True(void*);
extern PetscErrorCode PetscThreadInitialize_True(PetscInt);
extern PetscErrorCode PetscThreadFinalize_True(void);
extern void*          MainWait_True(void*);
						  extern PetscErrorCode MainJob_True(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* NO Thread Pool Functions */
extern void*          PetscThreadFunc_None(void*);
extern void*          MainWait_None(void*);
						  extern PetscErrorCode MainJob_None(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Lock free Functions */
extern void*          PetscThreadFunc_LockFree(void*);
extern PetscErrorCode PetscThreadInitialize_LockFree(PetscInt);
extern PetscErrorCode PetscThreadFinalize_LockFree(void);
extern void*           MainWait_LockFree(void*);
						  extern PetscErrorCode MainJob_LockFree(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

void* FuncFinish(void* arg) {
  PetscThreadGo = PETSC_FALSE;
  return(0);
}

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
/* Set CPU affinity for the main thread */
void PetscSetMainThreadAffinity(PetscInt icorr)
{
  cpu_set_t mset;
  int ncorr = get_nprocs();

  CPU_ZERO(&mset);
  CPU_SET(icorr%ncorr,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);
}

/* Set CPU affinity for individual threads */
void PetscPthreadSetAffinity(PetscInt icorr)
{
  cpu_set_t mset;
  int ncorr = get_nprocs();

  CPU_ZERO(&mset);
  CPU_SET(icorr%ncorr,&mset);
  pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mset);
}

void DoCoreAffinity(void)
{
  if (!PetscCheckCoreAffinity) return;
  else {
    int       i,icorr=0; 
    cpu_set_t mset;
    pthread_t pThread = pthread_self();

    for (i=0; i<PetscMaxThreads; i++) {
      if (pthread_equal(pThread,PetscThreadPoint[i])) {
        icorr = ThreadCoreAffinity[i];
	CPU_ZERO(&mset);
	CPU_SET(icorr,&mset);
	pthread_setaffinity_np(pThread,sizeof(cpu_set_t),&mset);
      }
    }
  }
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscSetMaxPThreads"
/*
   PetscSetMaxPThreads - Sets the number of pthreads to create.

   Not collective
  
   Input Parameters:
.  nthreads - # of pthreads.

   Options Database Keys:
   -nthreads <nthreads> Number of pthreads to create.

   Level: beginner
 
   Notes:
   Use nthreads = PETSC_DECIDE for PETSc to calculate the maximum number of pthreads to create.
   The number of threads is then set to the number of processing units available
   for the system. By default, PETSc will set max. threads = # of processing units
   available - 1 (since we consider the main thread as also a worker thread). If the
   option -mainthread_no_share_work is used, then max. threads created = # of
   available processing units.
   
.seealso: PetscGetMaxPThreads()
*/ 
PetscErrorCode PetscSetMaxPThreads(PetscInt nthreads) 
{
  PetscErrorCode ierr;
  PetscBool      flg=PETSC_FALSE;

  PetscFunctionBegin;

  N_CORES=1; /* Default value if N_CORES cannot be found out */
  PetscMaxThreads = N_CORES;
  /* Find the number of cores */
#if defined(PETSC_HAVE_SCHED_CPU_SET_T) /* Linux */
    N_CORES = get_nprocs();
#elif defined(PETSC_HAVE_SYS_SYSCTL_H) /* MacOS, BSD */
    size_t   len = sizeof(N_CORES);
    ierr = sysctlbyname("hw.activecpu",&N_CORES,&len,NULL,0);CHKERRQ(ierr);
#endif

  if(nthreads == PETSC_DECIDE) {
    /* Check if run-time option is given */
    ierr = PetscOptionsGetInt(PETSC_NULL,"-nthreads",&PetscMaxThreads,&flg);CHKERRQ(ierr);
    if(!flg) {
      PetscMaxThreads = N_CORES - PetscMainThreadShareWork;
    } 
  } else PetscMaxThreads = nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGetMaxPThreads"
/*
   PetscGetMaxPThreads - Returns the number of pthreads created.

   Not collective
  
   Output Parameters:
.  nthreads - Number of pthreads created.

   Level: beginner
 
   Notes:
   Must call PetscSetMaxPThreads() before
   
.seealso: PetscSetMaxPThreads()
*/ 
PetscErrorCode PetscGetMaxPThreads(PetscInt *nthreads)
{
  PetscFunctionBegin;
  if(PetscMaxThreads < 0) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscSetMaxPThreads() first");
  } else {
    *nthreads = PetscMaxThreads;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsCheckInitial_Private_Pthread"
PetscErrorCode PetscOptionsCheckInitial_Private_Pthread(void)
{
  PetscErrorCode                 ierr;
  PetscBool                      flg1=PETSC_FALSE;
  ThreadSynchronizationType      thread_sync_type=THREADSYNC_NOPOOL;

  PetscFunctionBegin;

  /* Check to see if the user wants the main thread not to share work with the other threads */
  ierr = PetscOptionsHasName(PETSC_NULL,"-mainthread_no_share_work",&flg1);CHKERRQ(ierr);
  if(flg1) PetscMainThreadShareWork = 0;

  /*
      Set maximum number of threads
  */
  ierr = PetscSetMaxPThreads(PETSC_DECIDE);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-main",&flg1);CHKERRQ(ierr);
  if(flg1) {
    ierr = PetscOptionsGetInt(PETSC_NULL,"-main",&MainThreadCoreAffinity,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
    PetscSetMainThreadAffinity(MainThreadCoreAffinity);
#endif
  }
 
  /* Set default affinities for threads: each thread has an affinity to one core unless the PetscMaxThreads > N_CORES */
  ThreadCoreAffinity = (int*)malloc(PetscMaxThreads*sizeof(int));
  char tstr[9];
  char tbuf[2];
  PetscInt i;
  
  strcpy(tstr,"-thread");
  for(i=0;i<PetscMaxThreads;i++) {
    ThreadCoreAffinity[i] = i+PetscMainThreadShareWork;
    sprintf(tbuf,"%d",i);
    strcat(tstr,tbuf);
    ierr = PetscOptionsHasName(PETSC_NULL,tstr,&flg1);CHKERRQ(ierr);
    if(flg1) {
      ierr = PetscOptionsGetInt(PETSC_NULL,tstr,&ThreadCoreAffinity[i],PETSC_NULL);CHKERRQ(ierr);
      ThreadCoreAffinity[i] = ThreadCoreAffinity[i]%N_CORES; /* check on the user */
    }
    tstr[7] = '\0';
  }
  
  PetscCheckCoreAffinity = PETSC_TRUE;

  ierr = PetscOptionsEnum("-thread_sync_type","Type of thread synchronization algorithm"," ",ThreadSynchronizationTypes,(PetscEnum)thread_sync_type,(PetscEnum*)&thread_sync_type,&flg1);CHKERRQ(ierr);
  
  switch(thread_sync_type) {
  case THREADSYNC_TREEPOOL:
    PetscThreadFunc       = &PetscThreadFunc_Tree;
    PetscThreadInitialize = &PetscThreadInitialize_Tree;
    PetscThreadFinalize   = &PetscThreadFinalize_Tree;
    MainWait              = &MainWait_Tree;
    MainJob               = &MainJob_Tree;
    PetscInfo1(PETSC_NULL,"Using tree thread pool with %d threads\n",PetscMaxThreads);
    break;
  case THREADSYNC_MAINPOOL:
    PetscThreadFunc       = &PetscThreadFunc_Main;
    PetscThreadInitialize = &PetscThreadInitialize_Main;
    PetscThreadFinalize   = &PetscThreadFinalize_Main;
    MainWait              = &MainWait_Main;
    MainJob               = &MainJob_Main;
    PetscInfo1(PETSC_NULL,"Using main thread pool with %d threads\n",PetscMaxThreads);
    break;
  case THREADSYNC_CHAINPOOL:
    PetscThreadFunc       = &PetscThreadFunc_Chain;
    PetscThreadInitialize = &PetscThreadInitialize_Chain;
    PetscThreadFinalize   = &PetscThreadFinalize_Chain;
    MainWait              = &MainWait_Chain;
    MainJob               = &MainJob_Chain;
    PetscInfo1(PETSC_NULL,"Using chain thread pool with %d threads\n",PetscMaxThreads);
    break;
  case THREADSYNC_TRUEPOOL:
#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
    PetscThreadFunc       = &PetscThreadFunc_True;
    PetscThreadInitialize = &PetscThreadInitialize_True;
    PetscThreadFinalize   = &PetscThreadFinalize_True;
    MainWait              = &MainWait_True;
    MainJob               = &MainJob_True;
    PetscInfo1(PETSC_NULL,"Using true thread pool with %d threads\n",PetscMaxThreads);
#else
    PetscThreadFunc       = &PetscThreadFunc_Main;
    PetscThreadInitialize = &PetscThreadInitialize_Main;
    PetscThreadFinalize   = &PetscThreadFinalize_Main;
    MainWait              = &MainWait_Main;
    MainJob               = &MainJob_Main;
    PetscInfo1(PETSC_NULL,"Cannot use true thread pool since pthread_barrier_t is not defined, creating main thread pool instead with %d threads\n",PetscMaxThreads);
#endif
    break;
  case THREADSYNC_NOPOOL:
    PetscThreadInitialize = PETSC_NULL;
    PetscThreadFinalize   = PETSC_NULL;
    PetscThreadFunc       = &PetscThreadFunc_None;
    MainWait              = &MainWait_None;
    MainJob               = &MainJob_None;
    PetscInfo1(PETSC_NULL,"Using No thread pool with %d threads\n",PetscMaxThreads);
    break;
  case THREADSYNC_LOCKFREE:
    PetscThreadFunc       = &PetscThreadFunc_LockFree;
    PetscThreadInitialize = &PetscThreadInitialize_LockFree;
    PetscThreadFinalize   = &PetscThreadFinalize_LockFree;
    MainWait              = &MainWait_LockFree;
    MainJob               = &MainJob_LockFree;
    PetscInfo1(PETSC_NULL,"Using lock-free algorithm with %d threads\n",PetscMaxThreads);
    break;
  }
  PetscFunctionReturn(0);
}
