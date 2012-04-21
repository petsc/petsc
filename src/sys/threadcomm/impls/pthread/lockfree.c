#include <../src/sys/threadcomm/impls/pthread/pthreadimpl.h>

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

void* PetscPThreadCommFunc_LockFree(void* arg)
{

#if defined(PETSC_PTHREAD_LOCAL)
  PetscPThreadRank = *(PetscInt*)arg;
#else
  PetscInt PetscPThreadRank=*(PetscInt*)arg;
  pthread_setspecific(PetscPThreadRankkey,&PetscPThreadRank);
#endif

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscPThreadCommDoCoreAffinity();
#endif

  /* Spin loop */
  while(PetscReadOnce(int,job_lockfree.my_job_status[PetscPThreadRank]) != -1) {
    if(job_lockfree.my_job_status[PetscPThreadRank] == 1) {
      (*job_lockfree.pfunc)(job_lockfree.pdata[PetscPThreadRank]);
      job_lockfree.my_job_status[PetscPThreadRank] = 0;
    }
  }

  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommInitialize_LockFree"
PetscErrorCode PetscPThreadCommInitialize_LockFree(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscThreadComm_PThread *ptcomm=(PetscThreadComm_PThread*)tcomm->data;

  PetscFunctionBegin;

  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(void*),&(job_lockfree.pdata));CHKERRQ(ierr);
  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&job_lockfree.my_job_status);CHKERRQ(ierr);
  ierr = PetscMalloc(ptcomm->nthreads*sizeof(PetscInt),&busy_threads.list);CHKERRQ(ierr);

  /* Create threads */
  for(i=ptcomm->thread_num_start; i < tcomm->nworkThreads;i++) {
    job_lockfree.pdata[i] = NULL;
    job_lockfree.my_job_status[i] = 0;
    ierr = pthread_create(&ptcomm->tid[i],NULL,&PetscPThreadCommFunc_LockFree,&ptcomm->ranks[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommFinalize_LockFree"
PetscErrorCode PetscPThreadCommFinalize_LockFree(PetscThreadComm tcomm)
{
  PetscErrorCode           ierr;
  void*                    jstatus;
  PetscThreadComm_PThread *ptcomm=(PetscThreadComm_PThread*)tcomm->data;
  PetscInt                 i;
  PetscFunctionBegin;
  for(i=ptcomm->thread_num_start; i < tcomm->nworkThreads;i++) {
    job_lockfree.my_job_status[i] = -1;
    ierr = pthread_join(ptcomm->tid[i],&jstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(job_lockfree.my_job_status);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.pdata);CHKERRQ(ierr);
  ierr = PetscFree(busy_threads.list);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommWait_LockFree"
PetscErrorCode PetscPThreadCommWait_LockFree(void)
{
  PetscInt active_threads=0,i;
  PetscBool wait=PETSC_TRUE;

  PetscFunctionBegin;
  /* Loop till all threads signal that they have done their job */
  while(wait) {
    for(i=0;i<busy_threads.nthreads;i++) active_threads += job_lockfree.my_job_status[busy_threads.list[i]];
    if(active_threads) active_threads = 0;
    else wait=PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommRunKernel_LockFree"
PetscErrorCode PetscPThreadCommRunKernel_LockFree(PetscThreadComm tcomm,PetscErrorCode (*pFunc)(void*),void** pdata)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread *ptcomm=(PetscThreadComm_PThread*)tcomm->data;
  PetscInt                i,thread_num,k=0;
  PetscFunctionBegin;
  job_lockfree.pfunc = pFunc;
  busy_threads.nthreads = tcomm->nworkThreads - ptcomm->thread_num_start;
  for(i=ptcomm->thread_num_start; i < tcomm->nworkThreads;i++) {
    thread_num = ptcomm->ranks[i];
    job_lockfree.pdata[thread_num] = pdata[i];
    busy_threads.list[k++] = thread_num;
    job_lockfree.my_job_status[thread_num] = 1;
  }
  if(ptcomm->ismainworker) (*pFunc)(pdata[0]);

  ierr = PetscPThreadCommWait_LockFree();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
