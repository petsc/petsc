
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

extern PetscBool    PetscThreadGo;              /* Flag to keep the threads spinning in a loop */
extern PetscMPIInt  PetscMaxThreads;            /* Max. threads created */
extern pthread_t*   PetscThreadPoint;           /* Pointer to thread ids */
extern int*         ThreadCoreAffinity;         /* Core affinity of each thread */
extern PetscInt     PetscMainThreadShareWork;   /* Is the main thread also a worker? 1 = Yes */
extern PetscInt     MainThreadCoreAffinity;     /* Core affinity of the main thread */


typedef enum {THREADSYNC_NOPOOL,THREADSYNC_MAINPOOL,THREADSYNC_TRUEPOOL,THREADSYNC_CHAINPOOL,THREADSYNC_TREEPOOL,THREADSYNC_LOCKFREE} ThreadSynchronizationType;
extern const char *const ThreadSynchronizationTypes[];

typedef enum {THREADAFFINITYPOLICY_ALL,THREADAFFINITYPOLICY_ONECORE} ThreadAffinityPolicyType;
extern const char *const ThreadAffinityPolicyTypes[];

/* Base function pointers */
extern void*          (*PetscThreadFunc)(void*);
extern PetscErrorCode (*PetscThreadInitialize)(PetscInt);
extern PetscErrorCode (*PetscThreadFinalize)(void);
extern void*          (*PetscThreadsWait)(void*);
extern PetscErrorCode (*PetscThreadsRunKernel)(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
extern void DoCoreAffinity(void);
#else
#define DoCoreAffinity()
#endif
extern void* FuncFinish(void*);

/* Tree Thread Pool Functions */
extern void*          PetscThreadFunc_Tree(void*);
extern PetscErrorCode PetscThreadInitialize_Tree(PetscInt);
extern PetscErrorCode PetscThreadFinalize_Tree(void);
extern void*          PetscThreadsWait_Tree(void*);
extern PetscErrorCode PetscThreadsRunKernel_Tree(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Main Thread Pool Functions */
extern void*          PetscThreadFunc_Main(void*);
extern PetscErrorCode PetscThreadInitialize_Main(PetscInt);
extern PetscErrorCode PetscThreadFinalize_Main(void);
extern void*          PetscThreadsWait_Main(void*);
extern PetscErrorCode PetscThreadsRunKernel_Main(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Chain Thread Pool Functions */
extern void*          PetscThreadFunc_Chain(void*);
extern PetscErrorCode PetscThreadInitialize_Chain(PetscInt);
extern PetscErrorCode PetscThreadFinalize_Chain(void);
extern void*          PetscThreadsWait_Chain(void*);
extern PetscErrorCode PetscThreadsRunKernel_Chain(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* True Thread Pool Functions */
extern void*          PetscThreadFunc_True(void*);
extern PetscErrorCode PetscThreadInitialize_True(PetscInt);
extern PetscErrorCode PetscThreadFinalize_True(void);
extern void*          PetscThreadsWait_True(void*);
extern PetscErrorCode PetscThreadsRunKernel_True(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* NO Thread Pool Functions */
extern void*          PetscThreadFunc_None(void*);
extern void*          PetscThreadsWait_None(void*);
extern PetscErrorCode PetscThreadsRunKernel_None(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

/* Lock free Functions */
extern void*          PetscThreadFunc_LockFree(void*);
extern PetscErrorCode PetscThreadInitialize_LockFree(PetscInt);
extern PetscErrorCode PetscThreadFinalize_LockFree(void);
extern void*          PetscThreadsWait_LockFree(void*);
extern PetscErrorCode PetscThreadsRunKernel_LockFree(void* (*pFunc)(void*),void**,PetscInt,PetscInt*);

#endif
