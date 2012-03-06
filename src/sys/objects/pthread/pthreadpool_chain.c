
#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

PetscInt*           pVal_chain;

#define CACHE_LINE_SIZE 64
extern PetscInt* ThreadCoreAffinity;

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

/*
 -----------------------------
    'Chain' Thread Functions 
 -----------------------------
*/
void* PetscThreadFunc_Chain(void* arg) {
  PetscInt* pId = (PetscInt*)arg;
  PetscInt ThreadId = *pId;
  PetscInt SubWorker = ThreadId + 1;
  PetscBool PeeOn;

  pthread_setspecific(rankkey,&threadranks[ThreadId+1]);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  DoCoreAffinity();
#endif
  if(ThreadId==(PetscMaxThreads-1)) {
    PeeOn = PETSC_TRUE;
  }
  else {
    PeeOn = PETSC_FALSE;
  }
  if(PeeOn==PETSC_FALSE) {
    /* check your subordinate, wait for him to be ready */
    pthread_mutex_lock(job_chain.mutexarray[SubWorker]);
    while(*(job_chain.arrThreadReady[SubWorker])==PETSC_FALSE) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
      pthread_cond_wait(job_chain.cond1array[SubWorker],job_chain.mutexarray[SubWorker]);
    }
    pthread_mutex_unlock(job_chain.mutexarray[SubWorker]);
    /* your subordinate is now ready*/
  }
  pthread_mutex_lock(job_chain.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_chain.arrThreadReady[ThreadId]) = PETSC_TRUE;
  if(ThreadId==0) {
    job_chain.eJobStat = JobCompleted;
    /* signal main */
    pthread_cond_signal(&main_cond_chain);
  }
  else {
    /* tell your boss that you're ready to work */
    pthread_cond_signal(job_chain.cond1array[ThreadId]);
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
        pthread_cond_wait(job_chain.cond2array[ThreadId],job_chain.mutexarray[ThreadId]);
	*(job_chain.arrThreadStarted[ThreadId]) = PETSC_TRUE;
	*(job_chain.arrThreadReady[ThreadId])   = PETSC_FALSE;
    }
    if(ThreadId==0) {
      job_chain.startJob = PETSC_FALSE;
      job_chain.eJobStat = ThreadsWorking;
    }
    pthread_mutex_unlock(job_chain.mutexarray[ThreadId]);
    if(PeeOn==PETSC_FALSE) {
      /* tell your subworker it's time to get to work */
      pthread_cond_signal(job_chain.cond2array[SubWorker]);
    }
    /* do your job */
    if(job_chain.funcArr[ThreadId+PetscMainThreadShareWork]) {
      job_chain.funcArr[ThreadId+PetscMainThreadShareWork](job_chain.pdata[ThreadId+PetscMainThreadShareWork]);
    }

    if(PetscThreadGo) {
      /* reset job, get ready for more */
      if(PeeOn==PETSC_FALSE) {
        /* check your subordinate, wait for him to be ready
         how do you know for a fact that your subordinate has actually started? */
        pthread_mutex_lock(job_chain.mutexarray[SubWorker]);
        while(*(job_chain.arrThreadReady[SubWorker])==PETSC_FALSE||*(job_chain.arrThreadStarted[SubWorker])==PETSC_FALSE) {
          /* upon entry, automically releases the lock and blocks
           upon return, has the lock */
          pthread_cond_wait(job_chain.cond1array[SubWorker],job_chain.mutexarray[SubWorker]);
        }
        pthread_mutex_unlock(job_chain.mutexarray[SubWorker]);
        /* your subordinate is now ready */
      }
      pthread_mutex_lock(job_chain.mutexarray[ThreadId]);
      *(job_chain.arrThreadReady[ThreadId]) = PETSC_TRUE;
      if(ThreadId==0) {
	job_chain.eJobStat = JobCompleted; /* foreman: last thread to complete, guaranteed! */
        /* root thread (foreman) signals 'main' */
        pthread_cond_signal(&main_cond_chain);
      }
      else {
        /* signal your boss before you go to sleep */
        pthread_cond_signal(job_chain.cond1array[ThreadId]);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationInitialize_Chain"
PetscErrorCode PetscThreadsSynchronizationInitialize_Chain(PetscInt N) 
{
  PetscErrorCode ierr;
  PetscInt i;

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
    ierr = pthread_mutex_init(job_chain.mutexarray[i],NULL);CHKERRQ(ierr);
    ierr = pthread_cond_init(job_chain.cond1array[i],NULL);CHKERRQ(ierr);
    ierr = pthread_cond_init(job_chain.cond2array[i],NULL);CHKERRQ(ierr);
    *(job_chain.arrThreadStarted[i])  = PETSC_FALSE;
    *(job_chain.arrThreadReady[i])    = PETSC_FALSE;
  }
  job_chain.funcArr = (pfunc*)malloc((N+PetscMainThreadShareWork)*sizeof(pfunc));
  job_chain.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  job_chain.startJob = PETSC_FALSE;
  job_chain.eJobStat = JobInitiated;
  pVal_chain = (PetscInt*)malloc(N*sizeof(PetscInt));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));

  threadranks[0] = 0;
  pthread_setspecific(rankkey,&threadranks[0]);
  /* create threads */
  for(i=0; i<N; i++) {
    pVal_chain[i] = i;
    threadranks[i+1] = i+1;
    job_chain.funcArr[i+PetscMainThreadShareWork] = NULL;
    job_chain.pdata[i+PetscMainThreadShareWork] = NULL;
    ierr = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_chain[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationFinalize_Chain"
PetscErrorCode PetscThreadsSynchronizationFinalize_Chain() {
  PetscInt       i;
  PetscErrorCode ierr;
  void*          jstatus;

  PetscFunctionBegin;

  PetscThreadsRunKernel(FuncFinish,NULL,PetscMaxThreads,PETSC_NULL);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);CHKERRQ(ierr);
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
#define __FUNCT__ "PetscThreadsWait_Chain"
void* PetscThreadsWait_Chain(void* arg) {
  
  pthread_mutex_lock(job_chain.mutexarray[0]);
  while(job_chain.eJobStat<JobCompleted||job_chain.startJob==PETSC_TRUE) {
    pthread_cond_wait(&main_cond_chain,job_chain.mutexarray[0]); 
  }
  pthread_mutex_unlock(job_chain.mutexarray[0]);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_Chain"
PetscErrorCode PetscThreadsRunKernel_Chain(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) {
  PetscInt i,j,issetaffinity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadsWait(NULL);
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
  ierr = pthread_cond_signal(job_chain.cond2array[0]);CHKERRQ(ierr);
  if(pFunc!=FuncFinish) {
    if(PetscMainThreadShareWork) {
      job_chain.funcArr[0] = pFunc;
      job_chain.pdata[0] = data[0];
      ierr = (PetscErrorCode)(long int)job_chain.funcArr[0](job_chain.pdata[0]);
    }
    PetscThreadsWait(NULL); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  PetscFunctionReturn(0);
}
