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
extern PetscInt     MainThreadCoreAffinity;

PetscErrorCode ithreaderr_lockfree = 0;
int*           pVal_lockfree;

extern int* ThreadCoreAffinity;

/* lock-free data structure */
typedef struct {
  void* (*pfunc)(void*);
  void** pdata;
  int *my_job_status;
} sjob_lockfree;
sjob_lockfree job_lockfree = {NULL,NULL,0};

/* external Functions */
extern void*          (*PetscThreadFunc)(void*);
extern PetscErrorCode (*PetscThreadInitialize)(PetscInt);
extern PetscErrorCode (*PetscThreadFinalize)(void);
extern void           (*MainWait)(void);
extern PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt);

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
extern void PetscPthreadSetAffinity(PetscInt);
extern void PetscSetMainThreadAffinity(PetscInt);
#endif

void* FuncFinish_LockFree(void* arg) {
  __sync_bool_compare_and_swap(&PetscThreadGo,PETSC_TRUE,PETSC_FALSE);
  return(0);
}

/* 
  ----------------------------
     'LockFree' Thread Functions 
  ----------------------------
*/
void* PetscThreadFunc_LockFree(void* arg) 
{
  int iVal;
  PetscErrorCode iterr;

  iVal = *(int*)arg;

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  int* pId      = (int*)arg;
  int  ThreadId = *pId; 
  PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);
#endif

  /* Spin loop */
  while(PetscThreadGo) {
    if(job_lockfree.my_job_status[iVal] == 0) {
      iterr = (PetscErrorCode)(long int)job_lockfree.pfunc(job_lockfree.pdata[iVal+PetscMainThreadShareWork]);
      __sync_bool_compare_and_swap(&job_lockfree.my_job_status[iVal],0,1);
    }
  }
  __sync_bool_compare_and_swap(&job_lockfree.my_job_status[iVal],0,1);
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_LockFree"
PetscErrorCode PetscThreadInitialize_LockFree(PetscInt N)
{
  PetscInt i,status;

  PetscFunctionBegin;
  pVal_lockfree = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  job_lockfree.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  job_lockfree.my_job_status = (int*)malloc(N*sizeof(int));

  /* Create threads */
  for(i=0; i<N; i++) {
    pVal_lockfree[i] = i;
    job_lockfree.my_job_status[i] = 1;
    job_lockfree.pdata[i] = NULL;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_lockfree[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_LockFree"
PetscErrorCode PetscThreadFinalize_LockFree() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  /* Signal all threads to finish */
  MainJob(FuncFinish_LockFree,NULL,PetscMaxThreads);

  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
  }

  free(job_lockfree.my_job_status);
  free(PetscThreadPoint);
  free(pVal_lockfree);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_LockFree"
void MainWait_LockFree() 
{
  int job_done,i;

  job_done = 0;
  /* Loop till all threads signal that they have done their job */
  while(!job_done) {
    for(i=0;i<PetscMaxThreads;i++)
      job_done += job_lockfree.my_job_status[i];
    if(job_done == PetscMaxThreads) job_done = 1;
    else job_done = 0;
  }
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_LockFree"
PetscErrorCode MainJob_LockFree(void* (*pFunc)(void*),void** data,PetscInt n) 
{
  int i;
  PetscErrorCode ijoberr = 0;

  job_lockfree.pfunc = pFunc;
  for(i=0;i<PetscMaxThreads;i++) {
    if(pFunc == FuncFinish_LockFree) job_lockfree.pdata[i+PetscMainThreadShareWork] = NULL;
    else job_lockfree.pdata = data;
    /* signal thread i to start the job */
    __sync_bool_compare_and_swap(&(job_lockfree.my_job_status[i]),1,0);
  }
  
  if(pFunc != FuncFinish_LockFree) {
    /* If the MainThreadShareWork flag is on then have the main thread also do the work */
    if(PetscMainThreadShareWork) {
      ijoberr = (PetscErrorCode)(long int)job_lockfree.pfunc(job_lockfree.pdata[0]);
    }
    /* Wait for all threads to finish their job */
    MainWait();
  }

  if(ithreaderr_lockfree) {
    ijoberr = ithreaderr_lockfree;
  }
  return ijoberr;
}
