
#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

/* lock-free data structure */
typedef struct {
  PetscErrorCode (*pfunc)(void*);
  void** pdata;
  PetscInt *my_job_status;
} sjob_lockfree;

static sjob_lockfree job_lockfree = {NULL,NULL,0};

/* This struct holds information for PetscThreadsWait_LockFree */
static struct {
  PetscInt nthreads; /* Number of busy threads */
  PetscInt *list;    /* List of busy threads */
} busy_threads;

#if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 1050
#define PetscAtomicCompareandSwap(ptr, oldval, newval) (OSAtomicCompareAndSwapPtr(oldval,newval,ptr))
#elif defined(_MSC_VER)
#define PetscAtomicCompareandSwap(ptr, oldval, newval) (InterlockedCompareExchange(ptr,newval,oldval))
#elif (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
#define PetscAtomicCompareandSwap(ptr, oldval, newval) (__sync_bool_compare_and_swap(ptr,oldval,newval))
#else
#  error No maping for PetscAtomicCompareandSwap
#endif

#define PetscReadOnce(type,val) (*(volatile type *)&val)
/* 
  ----------------------------
     'LockFree' Thread Functions 
  ----------------------------
*/
void* PetscThreadFunc_LockFree(void* arg)
{
#if defined(PETSC_PTHREAD_LOCAL)
  PetscThreadRank = *(PetscInt*)arg;
#else
  PetscInt PetscThreadRank=*(PetscInt*)arg;
  pthread_setspecific(PetscThreadsRankkey,&PetscThreadRank);
#endif

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscThreadsDoCoreAffinity();
#endif

  /* Spin loop */
  while(PetscReadOnce(int,job_lockfree.my_job_status[PetscThreadRank]) != -1) {
    if(job_lockfree.my_job_status[PetscThreadRank] == 1) {
      (*job_lockfree.pfunc)(job_lockfree.pdata[PetscThreadRank]);
      job_lockfree.my_job_status[PetscThreadRank] = 0;
    }
  }

  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationInitialize_LockFree"
PetscErrorCode PetscThreadsSynchronizationInitialize_LockFree(PetscInt N)
{
  PetscInt i;
  PetscErrorCode ierr;
  PetscInt nworkThreads=N+PetscMainThreadShareWork;

  PetscFunctionBegin;
  ierr = PetscMalloc(nworkThreads*sizeof(pthread_t),&PetscThreadPoint);CHKERRQ(ierr);
  ierr = PetscMalloc(nworkThreads*sizeof(void*),&(job_lockfree.pdata));CHKERRQ(ierr);
  ierr = PetscMalloc(nworkThreads*sizeof(PetscInt),&(job_lockfree.my_job_status));CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(PetscInt),&(busy_threads.list));CHKERRQ(ierr);

  if(PetscMainThreadShareWork) { 
    job_lockfree.pdata[0] =NULL;
    PetscThreadPoint[0] = pthread_self();
  }

  /* Create threads */
  for(i=PetscMainThreadShareWork; i<nworkThreads; i++) {
    job_lockfree.my_job_status[i] = 0;
    job_lockfree.pdata[i] = NULL;
    ierr = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&PetscThreadRanks[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSynchronizationFinalize_LockFree"
PetscErrorCode PetscThreadsSynchronizationFinalize_LockFree() 
{
  PetscInt       i;
  void*          jstatus;
  PetscErrorCode ierr;
  PetscInt       nworkThreads=PetscMaxThreads+PetscMainThreadShareWork;

  PetscFunctionBegin;

  /* join the threads */
  for(i=PetscMainThreadShareWork; i < nworkThreads; i++) {
    job_lockfree.my_job_status[i] = -1;
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);
  }

  ierr = PetscFree(PetscThreadPoint);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.my_job_status);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.pdata);CHKERRQ(ierr);
  ierr = PetscFree(busy_threads.list);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsWait_LockFree"
void* PetscThreadsWait_LockFree(void* arg) 
{
  PetscInt active_threads=0,i;
  PetscBool wait=PETSC_TRUE;

  /* Loop till all threads signal that they have done their job */
  while(wait) {
    for(i=0;i<busy_threads.nthreads;i++) active_threads += job_lockfree.my_job_status[busy_threads.list[i]];
    if(active_threads) active_threads = 0;
    else wait=PETSC_FALSE;
  }
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsRunKernel_LockFree"
PetscErrorCode PetscThreadsRunKernel_LockFree(PetscErrorCode (*pFunc)(void*),void** data,PetscInt n,PetscInt* cpu_affinity) 
{
  PetscInt i,j,k=0;
  PetscInt nworkThreads=PetscMaxThreads+PetscMainThreadShareWork;

  PetscFunctionBegin;

  busy_threads.nthreads = n-PetscMainThreadShareWork;
  job_lockfree.pfunc = pFunc;
  for(i=PetscMainThreadShareWork; i < n;i++) {
    for(j=PetscMainThreadShareWork;j < nworkThreads;j++) {
      if(cpu_affinity[i] == PetscThreadsCoreAffinities[j]) {
	job_lockfree.pdata[j] = data[i];
	busy_threads.list[k++] = j;
	/* signal thread j to start the job */
	job_lockfree.my_job_status[j] = 1;
      }
    }
  } 

  if(PetscMainThreadShareWork) (*pFunc)(data[0]);

  /* Wait for all busy threads to finish their job */
  PetscThreadsWait(NULL);
  
  PetscFunctionReturn(0);
}
