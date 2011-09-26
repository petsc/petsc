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

#if defined(PETSC_USE_PTHREAD)
PetscBool    PetscUseThreadPool    = PETSC_FALSE;
PetscBool    PetscThreadGo         = PETSC_TRUE;
PetscMPIInt  PetscMaxThreads = 2;
pthread_t*   PetscThreadPoint;
#if defined(PETSC_HAVE_PTHREAD_BARRIER)
pthread_barrier_t* BarrPoint;   /* used by 'true' thread pool */
#endif
PetscErrorCode ithreaderr = 0;
int*         pVal;

#define CACHE_LINE_SIZE 64  /* used by 'chain', 'main','tree' thread pools */
int* ThreadCoreAffinity;

typedef enum {JobInitiated,ThreadsWorking,JobCompleted} estat;  /* used by 'chain','tree' thread pool */

/* Tree thread pool data structure */
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  void* (*pfunc)(void*);
  void** pdata;
  PetscBool startJob;
  estat eJobStat;
  PetscBool** arrThreadStarted;
  PetscBool** arrThreadReady;
} sjob_tree;
sjob_tree job_tree;

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

/* chain thread pool data structure */
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  void* (*pfunc)(void*);
  void** pdata;
  PetscBool startJob;
  estat eJobStat;
  PetscBool** arrThreadStarted;
  PetscBool** arrThreadReady;
} sjob_chain;
sjob_chain job_chain;

/* true thread pool data structure */
#if defined(PETSC_HAVE_PTHREAD_BARRIER)
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

pthread_cond_t  main_cond  = PTHREAD_COND_INITIALIZER;  /* used by 'true', 'chain','tree' thread pools */
char* arrmutex; /* used by 'chain','main','tree' thread pools */
char* arrcond1; /* used by 'chain','main','tree' thread pools */
char* arrcond2; /* used by 'chain','main','tree' thread pools */
char* arrstart; /* used by 'chain','main','tree' thread pools */
char* arrready; /* used by 'chain','main','tree' thread pools */

/* Function Pointers */
void*          (*PetscThreadFunc)(void*) = NULL;
void*          (*PetscThreadInitialize)(PetscInt) = NULL;
PetscErrorCode (*PetscThreadFinalize)(void) = NULL;
void           (*MainWait)(void) = NULL;
PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt) = NULL;
/* Tree Thread Pool Functions */
void*          PetscThreadFunc_Tree(void*);
void*          PetscThreadInitialize_Tree(PetscInt);
PetscErrorCode PetscThreadFinalize_Tree(void);
void           MainWait_Tree(void);
PetscErrorCode MainJob_Tree(void* (*pFunc)(void*),void**,PetscInt);
/* Main Thread Pool Functions */
void*          PetscThreadFunc_Main(void*);
void*          PetscThreadInitialize_Main(PetscInt);
PetscErrorCode PetscThreadFinalize_Main(void);
void           MainWait_Main(void);
PetscErrorCode MainJob_Main(void* (*pFunc)(void*),void**,PetscInt);
/* Chain Thread Pool Functions */
void*          PetscThreadFunc_Chain(void*);
void*          PetscThreadInitialize_Chain(PetscInt);
PetscErrorCode PetscThreadFinalize_Chain(void);
void           MainWait_Chain(void);
PetscErrorCode MainJob_Chain(void* (*pFunc)(void*),void**,PetscInt);
/* True Thread Pool Functions */
void*          PetscThreadFunc_True(void*);
void*          PetscThreadInitialize_True(PetscInt);
PetscErrorCode PetscThreadFinalize_True(void);
void           MainWait_True(void);
PetscErrorCode MainJob_True(void* (*pFunc)(void*),void**,PetscInt);
/* NO Thread Pool Function */
PetscErrorCode MainJob_Spawn(void* (*pFunc)(void*),void**,PetscInt);

void* FuncFinish(void* arg) {
  PetscThreadGo = PETSC_FALSE;
  return(0);
}

PetscErrorCode PetscThreadRun(MPI_Comm Comm,void* (*funcp)(void*),int iTotThreads,pthread_t* ThreadId,void** data) 
{
  PetscErrorCode    ierr;
  PetscInt i;
  for(i=0; i<iTotThreads; i++) {
    ierr = pthread_create(&ThreadId[i],NULL,funcp,data[i]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscThreadStop(MPI_Comm Comm,int iTotThreads,pthread_t* ThreadId) 
{
  PetscErrorCode ierr;
  PetscInt i;

  PetscFunctionBegin;
  void* joinstatus;
  for (i=0; i<iTotThreads; i++) {
    ierr = pthread_join(ThreadId[i], &joinstatus);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*
  'Tree' Thread Pool Functions 
*/
void* PetscThreadFunc_Tree(void* arg) {
  PetscErrorCode iterr;
  int ierr;
  int* pId = (int*)arg;
  int ThreadId = *pId,Mary = 2,i,SubWorker;
  PetscBool PeeOn;
#if defined(PETSC_HAVE_CPU_SET_T)
  iterr = PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);CHKERRQ(iterr);
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
    ierr = pthread_cond_signal(&main_cond);
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
      iterr = (PetscErrorCode)(long int)job_tree.pfunc(job_tree.pdata[ThreadId]);
    }
    if(iterr!=0) {
      ithreaderr = 1;
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
        ierr = pthread_cond_signal(&main_cond);
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
void* PetscThreadInitialize_Tree(PetscInt N) {
  PetscInt i,ierr;
  int status;

  if(PetscUseThreadPool) {
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
    job_tree.pdata = (void**)malloc(N*sizeof(void*));
    job_tree.startJob = PETSC_FALSE;
    job_tree.eJobStat = JobInitiated;
    pVal = (int*)malloc(N*sizeof(int));
    /* allocate memory in the heap for the thread structure */
    PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
    /* create threads */
    for(i=0; i<N; i++) {
      pVal[i] = i;
      status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
      /* should check status */
    }
  }
  return NULL;
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
  free(pVal);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Tree"
void MainWait_Tree() {
  int ierr;
  ierr = pthread_mutex_lock(job_tree.mutexarray[0]);
  while(job_tree.eJobStat<JobCompleted||job_tree.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond,job_tree.mutexarray[0]);
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
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}

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

#if defined(PETSC_HAVE_CPU_SET_T)
    iterr = PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);CHKERRQ(iterr);
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
      iterr = (PetscErrorCode)(long int)job_main.pfunc(job_main.pdata[ThreadId]);
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
void* PetscThreadInitialize_Main(PetscInt N) {
  PetscInt i,ierr;
  int status;

  if(PetscUseThreadPool) {
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
    job_main.pdata = (void**)malloc(N*sizeof(void*));
    pVal = (int*)malloc(N*sizeof(int));
    /* allocate memory in the heap for the thread structure */
    PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
    /* create threads */
    for(i=0; i<N; i++) {
      pVal[i] = i;
      status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
      /* error check */
    }
  }
  else {
  }
  return NULL;
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
  free(pVal);

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
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}

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

#if defined(PETSC_HAVE_CPU_SET_T)
  iterr = PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);CHKERRQ(iterr);
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
    ierr = pthread_cond_signal(&main_cond);
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
    if(job_chain.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_chain.pfunc(job_chain.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_chain.pfunc(job_chain.pdata[ThreadId]);
    }
    if(iterr!=0) {
      ithreaderr = 1;
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
        ierr = pthread_cond_signal(&main_cond);
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
void* PetscThreadInitialize_Chain(PetscInt N) {
  PetscInt i,ierr;
  int status;

  if(PetscUseThreadPool) {
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
    job_chain.pfunc = NULL;
    job_chain.pdata = (void**)malloc(N*sizeof(void*));
    job_chain.startJob = PETSC_FALSE;
    job_chain.eJobStat = JobInitiated;
    pVal = (int*)malloc(N*sizeof(int));
    /* allocate memory in the heap for the thread structure */
    PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
    /* create threads */
    for(i=0; i<N; i++) {
      pVal[i] = i;
      status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
      /* should check error */
    }
  }
  else {
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_Chain"
PetscErrorCode PetscThreadFinalize_Chain() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
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
  free(pVal);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Chain"
void MainWait_Chain() {
  int ierr;
  ierr = pthread_mutex_lock(job_chain.mutexarray[0]);
  while(job_chain.eJobStat<JobCompleted||job_chain.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond,job_chain.mutexarray[0]);
  }
  ierr = pthread_mutex_unlock(job_chain.mutexarray[0]);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_Chain"
PetscErrorCode MainJob_Chain(void* (*pFunc)(void*),void** data,PetscInt n) {
  int i,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait();
  job_chain.pfunc = pFunc;
  job_chain.pdata = data;
  job_chain.startJob = PETSC_TRUE;
  for(i=0; i<PetscMaxThreads; i++) {
    *(job_chain.arrThreadStarted[i]) = PETSC_FALSE;
  }
  job_chain.eJobStat = JobInitiated;
  ierr = pthread_cond_signal(job_chain.cond2array[0]);
  if(pFunc!=FuncFinish) {
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}

#if defined(PETSC_HAVE_PTHREAD_BARRIER)
/* 
  ----------------------------
     'True' Thread Functions 
  ----------------------------
*/
void* PetscThreadFunc_True(void* arg) {
  int ierr,iVal;
  int* pId = (int*)arg;
  int ThreadId = *pId;
  PetscErrorCode iterr;

#if defined(PETSC_HAVE_CPU_SET_T)
  iterr = PetscPthreadSetAffinity(ThreadCoreAffinity[ThreadId]);CHKERRQ(iterr);
#endif
  ierr = pthread_mutex_lock(&job_true.mutex);
  job_true.iNumReadyThreads++;
  if(job_true.iNumReadyThreads==PetscMaxThreads) {
    ierr = pthread_cond_signal(&main_cond);
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
      ithreaderr = 1;
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
	ierr = pthread_cond_signal(&main_cond);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_True"
void* PetscThreadInitialize_True(PetscInt N) {
  PetscInt i;
  int status;

  pVal = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  BarrPoint = (pthread_barrier_t*)malloc((N+1)*sizeof(pthread_barrier_t)); /* BarrPoint[0] makes no sense, don't use it! */
  job_true.pdata = (void**)malloc(N*sizeof(void*));
  for(i=0; i<N; i++) {
    pVal[i] = i;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
    /* error check to ensure proper thread creation */
    status = pthread_barrier_init(&BarrPoint[i+1],NULL,i+1);
    /* should check error */
  }

  return NULL;
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

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_True"
void MainWait_True() {
  int ierr;
  ierr = pthread_mutex_lock(&job_true.mutex);
  while(job_true.iNumReadyThreads<PetscMaxThreads||job_true.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond,&job_true.mutex);
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

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}
#endif

/* 
   -----------------------------
     'NO' THREAD POOL FUNCTION 
   -----------------------------
*/
#undef __FUNCT__
#define __FUNCT__ "MainJob_Spawn"
PetscErrorCode MainJob_Spawn(void* (*pFunc)(void*),void** data,PetscInt n) {
  PetscErrorCode ijoberr = 0;

  pthread_t* apThread = (pthread_t*)malloc(n*sizeof(pthread_t));
  PetscThreadPoint = apThread; /* point to same place */
  PetscThreadRun(MPI_COMM_WORLD,pFunc,n,apThread,data);
  PetscThreadStop(MPI_COMM_WORLD,n,apThread); /* ensures that all threads are finished with the job */
  free(apThread);

  return ijoberr;
}

#if defined(PETSC_HAVE_CPU_SET_T)
PETSC_STATIC_INLINE PetscErrorCode PetscPthreadSchedSetAffinity(PetscInt icorr)
{
  cpu_set_t mset;
  int ncorr = get_nprocs();

  CPU_ZERO(&mset);
  CPU_SET(icorr%ncorr,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);
  return 0;
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsCheckInitial_Private_Pthread"
PetscErrorCode PetscOptionsCheckInitial_Private_Pthread(void)
{
  PetscErrorCode ierr;
  PetscBool flg1=PETSC_FALSE;

  PetscFunctionBegin;

  /*
      Determine whether user specified maximum number of threads
   */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-thread_max",&PetscMaxThreads,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-main",&flg1);CHKERRQ(ierr);
  if(flg1) {
    PetscInt icorr;
    ierr = PetscOptionsGetInt(PETSC_NULL,"-main",&icorr,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CPU_SET_T)
    ierr = PetscPthreadSetAffinity(icorr);CHKERRQ(ierr);
#endif
  }

#if defined(PETSC_HAVE_CPU_SET_T)
  PetscInt N_CORES = get_nprocs();
  ThreadCoreAffinity = (int*)malloc(N_CORES*sizeof(int));
  char tstr[9];
  char tbuf[2];
  PetscInt i;
  strcpy(tstr,"-thread");
  for(i=0;i<PetscMaxThreads;i++) {
    ThreadCoreAffinity[i] = i;
    sprintf(tbuf,"%d",i);
    strcat(tstr,tbuf);
    ierr = PetscOptionsHasName(PETSC_NULL,tstr,&flg1);CHKERRQ(ierr);
    if(flg1) {
      ierr = PetscOptionsGetInt(PETSC_NULL,tstr,&ThreadCoreAffinity[i],PETSC_NULL);CHKERRQ(ierr);
      ThreadCoreAffinity[i] = ThreadCoreAffinity[i]%N_CORES; /* check on the user */
    }
    tstr[7] = '\0';
  }
#endif

  /*
      Determine whether to use thread pool
   */
  ierr = PetscOptionsHasName(PETSC_NULL,"-use_thread_pool",&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscUseThreadPool = PETSC_TRUE;
    /* get the thread pool type */
    PetscInt ipool = 0;
    const char *choices[4] = {"true","tree","main","chain"};

    ierr = PetscOptionsGetEList(PETSC_NULL,"-use_thread_pool",choices,4,&ipool,PETSC_NULL);CHKERRQ(ierr);
    switch(ipool) {
    case 1:
      PetscThreadFunc       = &PetscThreadFunc_Tree;
      PetscThreadInitialize = &PetscThreadInitialize_Tree;
      PetscThreadFinalize   = &PetscThreadFinalize_Tree;
      MainWait              = &MainWait_Tree;
      MainJob               = &MainJob_Tree;
      PetscInfo(PETSC_NULL,"Using tree thread pool\n");
      break;
    case 2:
      PetscThreadFunc       = &PetscThreadFunc_Main;
      PetscThreadInitialize = &PetscThreadInitialize_Main;
      PetscThreadFinalize   = &PetscThreadFinalize_Main;
      MainWait              = &MainWait_Main;
      MainJob               = &MainJob_Main;
      PetscInfo(PETSC_NULL,"Using main thread pool\n");
      break;
    case 3:
      PetscThreadFunc       = &PetscThreadFunc_Chain;
      PetscThreadInitialize = &PetscThreadInitialize_Chain;
      PetscThreadFinalize   = &PetscThreadFinalize_Chain;
      MainWait              = &MainWait_Chain;
      MainJob               = &MainJob_Chain;
      PetscInfo(PETSC_NULL,"Using chain thread pool\n");
      break;
#if defined(PETSC_HAVE_PTHREAD_BARRIER)
    default:
      PetscThreadFunc       = &PetscThreadFunc_True;
      PetscThreadInitialize = &PetscThreadInitialize_True;
      PetscThreadFinalize   = &PetscThreadFinalize_True;
      MainWait              = &MainWait_True;
      MainJob               = &MainJob_True;
      PetscInfo(PETSC_NULL,"Using true thread pool\n");
      break;
# else
    default:
      PetscThreadFunc       = &PetscThreadFunc_Chain;
      PetscThreadInitialize = &PetscThreadInitialize_Chain;
      PetscThreadFinalize   = &PetscThreadFinalize_Chain;
      MainWait              = &MainWait_Chain;
      MainJob               = &MainJob_Chain;
      PetscInfo(PETSC_NULL,"Cannot use true thread pool since pthread_barrier_t is not available,using chain thread pool instead\n");
      break;
#endif
    }
    PetscThreadInitialize(PetscMaxThreads);
  } else {
    /* need to define these in the case on 'no threads' or 'thread create/destroy'
     could take any of the above versions 
    */
    MainJob               = &MainJob_Spawn;
  }

  PetscFunctionReturn(0);
}

#endif
