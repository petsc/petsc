#include <../src/sys/threadcomm/impls/pthread/pthreadimpl.h>

#define THREAD_TERMINATE      -1
#define THREAD_WAITING_FOR_JOB 0
#define THREAD_RECIEVED_JOB    1

/* lock-free data structure */
typedef struct {
  PetscThreadCommJobCtx *data;
  PetscInt           *my_job_status;
} sjob_lockfree;

static sjob_lockfree job_lockfree = {NULL,NULL};

static struct {
  PetscInt nthreads; /* Number of busy threads */
  PetscInt *list;    /* List of busy threads */
} busy_threads;

void RunJob(PetscInt nargs,PetscThreadCommJobCtx job)
{
  switch(nargs) {
  case 0:
    (*job->pfunc)(PetscPThreadRank);
    break;
  case 1:
    (*job->pfunc)(PetscPThreadRank,job->args[0]);
    break;
  case 2:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1]);
    break;
  case 3:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2]);
    break;
  case 4:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2],job->args[3]);
    break;
  case 5:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4]);
    break;
  case 6:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5]);
    break;
  case 7:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6]);
    break;
  case 8:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7]);
    break;
  case 9:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8]);
    break;
  case 10:
    (*job->pfunc)(PetscPThreadRank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8],job->args[9]);
    break;
  }
}

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
  while(PetscReadOnce(int,job_lockfree.my_job_status[PetscPThreadRank]) != THREAD_TERMINATE) {
    if(job_lockfree.my_job_status[PetscPThreadRank] == THREAD_RECIEVED_JOB) {
      RunJob(job_lockfree.data[PetscPThreadRank]->nargs,job_lockfree.data[PetscPThreadRank]);
      job_lockfree.my_job_status[PetscPThreadRank] = THREAD_WAITING_FOR_JOB;
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

  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscThreadCommJobCtx),&job_lockfree.data);CHKERRQ(ierr);
  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&job_lockfree.my_job_status);CHKERRQ(ierr);
  ierr = PetscMalloc(ptcomm->nthreads*sizeof(PetscInt),&busy_threads.list);CHKERRQ(ierr);

  /* Create threads */
  for(i=ptcomm->thread_num_start; i < tcomm->nworkThreads;i++) {
    job_lockfree.my_job_status[i] = THREAD_WAITING_FOR_JOB;
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
    job_lockfree.my_job_status[i] = THREAD_TERMINATE;
    ierr = pthread_join(ptcomm->tid[i],&jstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(job_lockfree.my_job_status);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.data);CHKERRQ(ierr);
  ierr = PetscFree(busy_threads.list);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommBarrier_LockFree"
PetscErrorCode PetscPThreadCommBarrier_LockFree(void)
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
PetscErrorCode PetscPThreadCommRunKernel_LockFree(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread *ptcomm=(PetscThreadComm_PThread*)tcomm->data;
  PetscInt                i,thread_num,k=0;
  PetscFunctionBegin;
  busy_threads.nthreads = tcomm->nworkThreads - ptcomm->thread_num_start;
  for(i=ptcomm->thread_num_start; i < tcomm->nworkThreads;i++) {
    thread_num = ptcomm->ranks[i];
    job_lockfree.data[thread_num] = job;
    busy_threads.list[k++] = thread_num;
    job_lockfree.my_job_status[thread_num] = THREAD_RECIEVED_JOB;
  }
  if(ptcomm->ismainworker) {
    job_lockfree.data[0] = job;
    RunJob(job->nargs, job_lockfree.data[0]);
  }

  ierr = PetscPThreadCommBarrier_LockFree();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
