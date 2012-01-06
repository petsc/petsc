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

extern pthread_t* PetscThreadPoint;
extern int*       ThreadCoreAffinity;

extern void*   (*PetscThreadFunc)(void*);
extern void*   (*MainWait)(void*);

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
  PetscErrorCode    iterr=0;
  sjob_none         *job = (sjob_none*)arg;
  PetscInt          i;
  PetscInt          nthreads= (int)job->nthreads;
  void**            data = (void**)job->data;
  pthread_t*        ThreadId = (pthread_t*)job->ThreadId;
  pfunc             funcp = (pfunc)job->kernelfunc;

  for(i=0; i<nthreads; i++) {
    iterr = pthread_create(&ThreadId[i],NULL,funcp,data[i]);
  }
  return(0);
}

void* MainWait_None(void* arg)
{
  sjob_none      *job=(sjob_none*)arg;
  int            nthreads = job->nthreads;
  pthread_t*     ThreadId = (pthread_t*)job->ThreadId;
  int            ierr;
  PetscInt       i;
  void*          joinstatus;
  for (i=0; i<nthreads; i++) {
    ierr = pthread_join(ThreadId[i], &joinstatus);
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_None"
PetscErrorCode MainJob_None(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) {
  PetscErrorCode ijoberr = 0,i;

  PetscFunctionBegin;
  pthread_t* apThread = (pthread_t*)malloc(n*sizeof(pthread_t));
  PetscThreadPoint = apThread; /* point to same place */
  job_none.nthreads = n;
  job_none.kernelfunc = pFunc;
  job_none.data = data;
  job_none.ThreadId = apThread;
  for(i=0;i<n;i++) ThreadCoreAffinity[i] = cpu_affinity[i];
  PetscThreadFunc(&job_none);
  MainWait(&job_none); /* ensures that all threads are finished with the job */
  free(apThread);

  PetscFunctionReturn(0);
}
