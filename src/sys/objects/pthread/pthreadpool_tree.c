
#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

static int*         pVal_tree;

#define CACHE_LINE_SIZE 64

typedef enum {JobInitiated,ThreadsWorking,JobCompleted} estat_tree;

typedef void* (*pfunc)(void*);

/* Tree thread pool data structure */
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  pfunc* funcArr;
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

/*
  'Tree' Thread Pool Functions 
*/
void* PetscThreadFunc_Tree(void* arg) 
{
  int* pId = (int*)arg;
  int ThreadId = *pId,Mary = 2,i,SubWorker;
  PetscBool PeeOn;

  pthread_setspecific(rankkey,&threadranks[ThreadId+1]);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  DoCoreAffinity();
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
        pthread_mutex_lock(job_tree.mutexarray[SubWorker]);
        while(*(job_tree.arrThreadReady[SubWorker])==PETSC_FALSE) {
          /* upon entry, automically releases the lock and blocks
           upon return, has the lock */
          pthread_cond_wait(job_tree.cond1array[SubWorker],job_tree.mutexarray[SubWorker]);
        }
        pthread_mutex_unlock(job_tree.mutexarray[SubWorker]);
      }
    }
    /* your subordinates are now ready */
  }
  pthread_mutex_lock(job_tree.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_tree.arrThreadReady[ThreadId]) = PETSC_TRUE;
  if(ThreadId==0) {
    job_tree.eJobStat = JobCompleted;
    /* signal main */
    pthread_cond_signal(&main_cond_tree);
  }
  else {
    /* tell your boss that you're ready to work */
    pthread_cond_signal(job_tree.cond1array[ThreadId]);
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
        pthread_cond_wait(job_tree.cond2array[ThreadId],job_tree.mutexarray[ThreadId]);
	*(job_tree.arrThreadStarted[ThreadId]) = PETSC_TRUE;
	*(job_tree.arrThreadReady[ThreadId])   = PETSC_FALSE;
    }
    if(ThreadId==0) {
      job_tree.startJob = PETSC_FALSE;
      job_tree.eJobStat = ThreadsWorking;
    }
    pthread_mutex_unlock(job_tree.mutexarray[ThreadId]);
    if(PeeOn==PETSC_FALSE) {
      /* tell your subordinates it's time to get to work */
      for(i=1; i<=Mary; i++) {
	SubWorker = Mary*ThreadId+i;
        if(SubWorker<PetscMaxThreads) {
          pthread_cond_signal(job_tree.cond2array[SubWorker]);
        }
      }
    }
    /* do your job */
    if(job_tree.funcArr[ThreadId+PetscMainThreadShareWork]) {
      job_tree.funcArr[ThreadId+PetscMainThreadShareWork](job_tree.pdata[ThreadId+PetscMainThreadShareWork]);
    }

    if(PetscThreadGo) {
      /* reset job, get ready for more */
      if(PeeOn==PETSC_FALSE) {
        /* check your subordinates, waiting for them to be ready
         how do you know for a fact that a given subordinate has actually started? */
	for(i=1;i<=Mary;i++) {
	  SubWorker = Mary*ThreadId+i;
          if(SubWorker<PetscMaxThreads) {
            pthread_mutex_lock(job_tree.mutexarray[SubWorker]);
            while(*(job_tree.arrThreadReady[SubWorker])==PETSC_FALSE||*(job_tree.arrThreadStarted[SubWorker])==PETSC_FALSE) {
              /* upon entry, automically releases the lock and blocks
               upon return, has the lock */
              pthread_cond_wait(job_tree.cond1array[SubWorker],job_tree.mutexarray[SubWorker]);
            }
            pthread_mutex_unlock(job_tree.mutexarray[SubWorker]);
          }
	}
        /* your subordinates are now ready */
      }
      pthread_mutex_lock(job_tree.mutexarray[ThreadId]);
      *(job_tree.arrThreadReady[ThreadId]) = PETSC_TRUE;
      if(ThreadId==0) {
	job_tree.eJobStat = JobCompleted; /* oot thread: last thread to complete, guaranteed! */
        /* root thread signals 'main' */
        pthread_cond_signal(&main_cond_tree);
      }
      else {
        /* signal your boss before you go to sleep */
        pthread_cond_signal(job_tree.cond1array[ThreadId]);
      }
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationInitialize_Tree"
PetscErrorCode PetscThreadsSynchronizationInitialize_Tree(PetscInt N) 
{
  PetscErrorCode ierr;
  PetscInt       i;

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
    ierr = pthread_mutex_init(job_tree.mutexarray[i],NULL);CHKERRQ(ierr);
    ierr = pthread_cond_init(job_tree.cond1array[i],NULL);CHKERRQ(ierr);
    ierr = pthread_cond_init(job_tree.cond2array[i],NULL);CHKERRQ(ierr);
    *(job_tree.arrThreadStarted[i])  = PETSC_FALSE;
    *(job_tree.arrThreadReady[i])    = PETSC_FALSE;
  }
  job_tree.funcArr = (pfunc*)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  job_tree.pdata = (void**)malloc((N+PetscMainThreadShareWork)*sizeof(void*));
  job_tree.startJob = PETSC_FALSE;
  job_tree.eJobStat = JobInitiated;
  pVal_tree = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  threadranks[0] = 0;
  pthread_setspecific(rankkey,&threadranks[0]);
  /* create threads */
  for(i=0; i<N; i++) {
    pVal_tree[i] = i;
    threadranks[i+1] = i+1;
    job_tree.funcArr[i+PetscMainThreadShareWork] = NULL;
    job_tree.pdata[i+PetscMainThreadShareWork] = NULL;
    ierr = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_tree[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationFinalize_Tree"
PetscErrorCode PetscThreadsSynchronizationFinalize_Tree() {
  PetscInt       i;
  void*          jstatus;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscThreadsRunKernel(FuncFinish,NULL,PetscMaxThreads,PETSC_NULL);  /* set up job and broadcast work */
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
  free(job_tree.funcArr);
  free(job_tree.pdata);
  free(pVal_tree);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsWait_Tree"
void* PetscThreadsWait_Tree(void* arg) {

  pthread_mutex_lock(job_tree.mutexarray[0]);
  while(job_tree.eJobStat<JobCompleted||job_tree.startJob==PETSC_TRUE) {
    pthread_cond_wait(&main_cond_tree,job_tree.mutexarray[0]);
  }
  pthread_mutex_unlock(job_tree.mutexarray[0]);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_Tree"
PetscErrorCode PetscThreadsRunKernel_Tree(void* (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) 
{
  int i,j,issetaffinity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadsWait(NULL);
  job_tree.startJob = PETSC_TRUE;
  for(i=0; i<PetscMaxThreads; i++) {
    if(pFunc == FuncFinish) {
      job_tree.funcArr[i+PetscMainThreadShareWork] = pFunc;
      job_tree.pdata[i+PetscMainThreadShareWork] = NULL;
    } else {
      issetaffinity=0;
      for(j=PetscMainThreadShareWork;j < n;j++) {
	if(cpu_affinity[j] == ThreadCoreAffinity[i]) {
	  job_tree.funcArr[i+PetscMainThreadShareWork] = pFunc;
	  job_tree.pdata[i+PetscMainThreadShareWork] = data[j];
	  issetaffinity=1;
	}
      }
      if(!issetaffinity) {
	job_tree.funcArr[i+PetscMainThreadShareWork] = NULL;
	job_tree.pdata[i+PetscMainThreadShareWork] = NULL;
      }
    }
    *(job_tree.arrThreadStarted[i]) = PETSC_FALSE;
  }
  job_tree.eJobStat = JobInitiated;
  ierr = pthread_cond_signal(job_tree.cond2array[0]);CHKERRQ(ierr);
  if(pFunc!=FuncFinish) {
    if(PetscMainThreadShareWork) {
      job_tree.funcArr[0] = pFunc;
      job_tree.pdata[0] = data[0];
      job_tree.funcArr[0](job_tree.pdata[0]);
    }
    PetscThreadsWait(NULL); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  PetscFunctionReturn(0);
}
