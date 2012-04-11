
#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

static PetscInt*           pVal_true;

#define CACHE_LINE_SIZE 64

typedef PetscErrorCode (*pfunc)(void*);

/* true thread pool data structure */
#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
pthread_barrier_t pbarr;
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pfunc *funcArr;
  void** pdata;
  PetscInt iNumJobThreads;
  PetscInt iNumReadyThreads;
  PetscBool startJob;
} sjob_true;

static sjob_true job_true = {PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER,NULL,NULL,0,0,PETSC_FALSE};
#endif

static pthread_cond_t main_cond_true = PTHREAD_COND_INITIALIZER; 

#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
/* 
  ----------------------------
     'True' Thread Functions 
  ----------------------------
*/
void* PetscThreadFunc_True(void* arg) 
{
  PetscInt* pId      = (PetscInt*)arg;
  PetscInt  ThreadId = *pId; 
  PetscThreadRank=ThreadId+1;

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscThreadsDoCoreAffinity();
#endif
  pthread_mutex_lock(&job_true.mutex);
  job_true.iNumReadyThreads++;
  if(job_true.iNumReadyThreads==PetscMaxThreads) {
    pthread_cond_signal(&main_cond_true);
  }
  /*the while loop needs to have an exit
    the 'main' thread can terminate all the threads by performing a broadcast
   and calling PetscThreadsFinish */
  while(PetscThreadGo) {
    /*need to check the condition to ensure we don't have to wait
      waiting when you don't have to causes problems
     also need to wait if another thread sneaks in and messes with the predicate */
    while(job_true.startJob==PETSC_FALSE&&job_true.iNumJobThreads==0) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
      pthread_cond_wait(&job_true.cond,&job_true.mutex);
    }
    job_true.startJob = PETSC_FALSE;
    job_true.iNumJobThreads--;
    job_true.iNumReadyThreads--;
    pthread_mutex_unlock(&job_true.mutex);

    if(job_true.funcArr[ThreadId+PetscMainThreadShareWork]) {
      job_true.funcArr[ThreadId+PetscMainThreadShareWork](job_true.pdata[ThreadId+PetscMainThreadShareWork]);
    }

    /* the barrier is necessary BECAUSE: look at job_true.iNumReadyThreads
      what happens if a thread finishes before they all start? BAD!
     what happens if a thread finishes before any else start? BAD! */
    pthread_barrier_wait(&pbarr); /* ensures all threads are finished */
    /* reset job */
    if(PetscThreadGo) {
      pthread_mutex_lock(&job_true.mutex);
      job_true.iNumReadyThreads++;
      if(job_true.iNumReadyThreads==PetscMaxThreads) {
	/* signal the 'main' thread that the job is done! (only done once) */
	pthread_cond_signal(&main_cond_true);
      }
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationInitialize_True"
PetscErrorCode PetscThreadsSynchronizationInitialize_True(PetscInt N)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;

  ierr = PetscMalloc(N*sizeof(PetscInt),&pVal_true);CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(pthread_t),&PetscThreadPoint);CHKERRQ(ierr);
  ierr = PetscMalloc((N+PetscMainThreadShareWork)*sizeof(pfunc),&(job_true.funcArr));CHKERRQ(ierr);
  ierr = PetscMalloc((N+PetscMainThreadShareWork)*sizeof(void*),&(job_true.pdata));CHKERRQ(ierr);

  /* Initialize the barrier */
  ierr = pthread_barrier_init(&pbarr,NULL,PetscMaxThreads);CHKERRQ(ierr);

  PetscThreadRank=0;
  for(i=0; i<N; i++) {
    pVal_true[i] = i;
    job_true.funcArr[i+PetscMainThreadShareWork] = NULL;
    job_true.pdata[i+PetscMainThreadShareWork] = NULL;
    ierr = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal_true[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationFinalize_True"
PetscErrorCode PetscThreadsSynchronizationFinalize_True() {
  PetscInt            i;
  void*          jstatus;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscThreadsRunKernel(PetscThreadsFinish,NULL,PetscMaxThreads,PETSC_NULL);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);
  }

  ierr = PetscFree(pVal_true);CHKERRQ(ierr);
  ierr = PetscFree(PetscThreadPoint);CHKERRQ(ierr);
  ierr = PetscFree(job_true.funcArr);CHKERRQ(ierr);
  ierr = PetscFree(job_true.pdata);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsWait_True"
void* PetscThreadsWait_True(void* arg) {

  pthread_mutex_lock(&job_true.mutex);
  while(job_true.iNumReadyThreads<PetscMaxThreads||job_true.startJob==PETSC_TRUE) {
    pthread_cond_wait(&main_cond_true,&job_true.mutex);
  }
  pthread_mutex_unlock(&job_true.mutex);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_True"
PetscErrorCode PetscThreadsRunKernel_True(PetscErrorCode (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) 
{
  PetscInt i,j,issetaffinity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadsWait(NULL);
  for(i=0; i<PetscMaxThreads; i++) {
    if(pFunc == PetscThreadsFinish) {
      job_true.funcArr[i+PetscMainThreadShareWork] = pFunc;
      job_true.pdata[i+PetscMainThreadShareWork] = NULL;
    } else {
      issetaffinity=0;
      for(j=PetscMainThreadShareWork;j < n;j++) {
	if(cpu_affinity[j] == PetscThreadsCoreAffinities[i]) {
	  job_true.funcArr[i+PetscMainThreadShareWork] = pFunc;
	  job_true.pdata[i+PetscMainThreadShareWork] = data[j];
	  issetaffinity=1;
	}
      }
      if(!issetaffinity) {
	job_true.funcArr[i+PetscMainThreadShareWork] = NULL;
	job_true.pdata[i+PetscMainThreadShareWork] = NULL;
      }

    }
  }

  job_true.iNumJobThreads = PetscMaxThreads;;
  job_true.startJob = PETSC_TRUE;
  /* Tell the threads to go to work */
  ierr = pthread_cond_broadcast(&job_true.cond);CHKERRQ(ierr);
  if(pFunc!=PetscThreadsFinish) {
    if(PetscMainThreadShareWork) {
      (*pFunc)(data[0]);
    }
    PetscThreadsWait(NULL); /* why wait after? guarantees that job gets done */
  }

  PetscFunctionReturn(0);
}
#else
PetscInt PetscPthread_dummy()
{
  return 0;
}
#endif
