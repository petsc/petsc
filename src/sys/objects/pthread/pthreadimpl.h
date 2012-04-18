
#if !defined(__PETSCTHREADS_H)
#define __PETSCTHREADS_H

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
#elif defined(PETSC_HAVE_WINPTHREADS_H)
#include "winpthreads.h"       /* http://locklessinc.com/downloads/winpthreads.h */
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

extern PetscBool      PetscThreadGo;                 /* Flag to keep the threads spinning in a loop */
extern PetscMPIInt    PetscMaxThreads;               /* Max. threads created */
extern pthread_t*     PetscThreadPoint;              /* Pointer to thread ids */
extern PetscInt*      PetscThreadsCoreAffinities;    /* Core affinity of threads, includes the main thread affinity 
                                                        if the main thread is also a work thread */
extern PetscInt       PetscMainThreadShareWork;      /* Is the main thread also a worker? 1 = Yes (Default)*/
extern PetscInt       PetscMainThreadCoreAffinity;   /* Core affinity of the main thread */
extern PetscBool      PetscThreadsInitializeCalled;  /* Check whether PetscThreadsInitialize has been called */ 
#if defined(PETSC_PTHREAD_LOCAL)
extern PETSC_PTHREAD_LOCAL PetscInt PetscThreadRank; /* Rank of the thread ... thread local variable */
#else
extern pthread_key_t  PetscThreadsRankkey;
#endif
extern PetscInt*      PetscThreadRanks;              /* Thread ranks - if main thread is a worker then main thread 
                                                        rank is 0 and ranks for other threads start from 1, 
                                                        otherwise the thread ranks start from 0 */

/*
  PetscThreadsSynchronizationType - Type of thread synchronization for pthreads

$ THREADSYNC_NOPOOL   - threads created and destroyed as and when required.
  The following synchronization scheme uses a thread pool mechanism and uses 
  mutex and condition variables for thread synchronization.
$ THREADSYNC_MAINPOOL  - Individual threads are woken up sequentially by main thread.
$ THREADSYNC_TRUEPOOL  - Uses a broadcast to wake up all threads
$ THREADSYNC_CHAINPOOL - Uses a chain scheme for waking up threads.
$ THREADSYNC_TREEPOOL  - Uses a tree scheme for waking up threads.
  The following synchronization scheme uses GCC's atomic function __sync_bool_compare_and_swap
  for thread synchronization
$ THREADSYNC_LOCKFREE -  A lock-free variant of the MAINPOOL model.

  Level: developer
*/
typedef enum {THREADSYNC_NOPOOL,THREADSYNC_MAINPOOL,THREADSYNC_TRUEPOOL,THREADSYNC_CHAINPOOL,THREADSYNC_TREEPOOL,THREADSYNC_LOCKFREE} PetscThreadsSynchronizationType;

/*
  PetscThreadsAffinityPolicyType - Core affinity policy for pthreads

$ THREADAFFINITYPOLICY_ALL - threads can run on any core.
$ THREADAFFINITYPOLICY_ONECORE - threads can run on only one core
$ THREADAFFINITYPOLICY_NONE - No affinity policy
   Level: developer
*/
typedef enum {THREADAFFINITYPOLICY_ALL,THREADAFFINITYPOLICY_ONECORE,THREADAFFINITYPOLICY_NONE} PetscThreadsAffinityPolicyType;

/* Base function pointers */
extern void*          (*PetscThreadFunc)(void*);
extern PetscErrorCode (*PetscThreadsSynchronizationInitialize)(PetscInt);
extern PetscErrorCode (*PetscThreadsSynchronizationFinalize)(void);
extern void*          (*PetscThreadsWait)(void*);
extern PetscErrorCode (*PetscThreadsRunKernel)(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
extern void PetscThreadsDoCoreAffinity(void);
#else
#define PetscThreadsDoCoreAffinity()
#endif
extern PetscErrorCode PetscThreadsFinish(void*);

extern PetscErrorCode PetscThreadsInitialize(PetscInt);
extern PetscErrorCode PetscThreadsFinalize(void);

#if 0 /* Disabling all the thread thread pools except lockfree */
/* Tree Thread Pool Functions */
extern void*          PetscThreadFunc_Tree(void*);
extern PetscErrorCode PetscThreadsSynchronizationInitialize_Tree(PetscInt);
extern PetscErrorCode PetscThreadsSynchronizationFinalize_Tree(void);
extern void*          PetscThreadsWait_Tree(void*);
extern PetscErrorCode PetscThreadsRunKernel_Tree(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Main Thread Pool Functions */
extern void*          PetscThreadFunc_Main(void*);
extern PetscErrorCode PetscThreadsSynchronizationInitialize_Main(PetscInt);
extern PetscErrorCode PetscThreadsSynchronizationFinalize_Main(void);
extern void*          PetscThreadsWait_Main(void*);
extern PetscErrorCode PetscThreadsRunKernel_Main(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Chain Thread Pool Functions */
extern void*          PetscThreadFunc_Chain(void*);
extern PetscErrorCode PetscThreadsSynchronizationInitialize_Chain(PetscInt);
extern PetscErrorCode PetscThreadsSynchronizationFinalize_Chain(void);
extern void*          PetscThreadsWait_Chain(void*);
extern PetscErrorCode PetscThreadsRunKernel_Chain(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* True Thread Pool Functions */
extern void*          PetscThreadFunc_True(void*);
extern PetscErrorCode PetscThreadsSynchronizationInitialize_True(PetscInt);
extern PetscErrorCode PetscThreadsSynchronizationFinalize_True(void);
extern void*          PetscThreadsWait_True(void*);
extern PetscErrorCode PetscThreadsRunKernel_True(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* NO Thread Pool Functions */
extern void*          PetscThreadFunc_None(void*);
extern void*          PetscThreadsWait_None(void*);
extern PetscErrorCode PetscThreadsRunKernel_None(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*);
#endif

/* Lock free Functions */
extern void*          PetscThreadFunc_LockFree(void*);
extern PetscErrorCode PetscThreadsSynchronizationInitialize_LockFree(PetscInt);
extern PetscErrorCode PetscThreadsSynchronizationFinalize_LockFree(void);
extern void*          PetscThreadsWait_LockFree(void*);
extern PetscErrorCode PetscThreadsRunKernel_LockFree(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*);

#endif
