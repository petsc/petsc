#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#define THREAD_TERMINATE      -1
#define THREAD_WAITING_FOR_JOB 0
#define THREAD_RECIEVED_JOB    1
#define THREAD_INITIALIZE      2

/* lock-free data structure */
typedef struct {
  PetscThreadCommJobCtx *data;
  PetscInt           *my_job_status;
} sjob_lockfree;

static sjob_lockfree job_lockfree = {NULL,NULL};


void SparkThreads_LockFree(PetscInt myrank,PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscInt i,thread_num;
  PetscThreadComm_PThread  ptcomm;
  PetscInt                 next;

  ptcomm = (PetscThreadComm_PThread)tcomm->data;
  
  switch(ptcomm->spark) {
  case PTHREADPOOLSPARK_LEADER:
    if(PetscReadOnce(int,tcomm->leader) == myrank) {
      /* Only leader sparks all the other threads */
      for(i=ptcomm->thread_num_start+1; i < tcomm->nworkThreads;i++) {
	thread_num = ptcomm->granks[i];
	while(PetscReadOnce(int,job_lockfree.my_job_status[thread_num]) != THREAD_WAITING_FOR_JOB)
	  ;
	job_lockfree.data[thread_num] = job;
	job_lockfree.my_job_status[thread_num] = THREAD_RECIEVED_JOB;
      }
    }
    break;
  case PTHREADPOOLSPARK_CHAIN:
    /* Spark the next thread */
    next = ptcomm->ngranks[myrank];
    if(next != -1) {
      thread_num = next;
      while(PetscReadOnce(int,job_lockfree.my_job_status[thread_num]) != THREAD_WAITING_FOR_JOB)
	;
      job_lockfree.data[thread_num] = job;
      job_lockfree.my_job_status[thread_num] = THREAD_RECIEVED_JOB;
    }
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
  job_lockfree.my_job_status[PetscPThreadRank] = THREAD_WAITING_FOR_JOB;

  /* Spin loop */
  while(PetscReadOnce(int,job_lockfree.my_job_status[PetscPThreadRank]) != THREAD_TERMINATE) {
    if(job_lockfree.my_job_status[PetscPThreadRank] == THREAD_RECIEVED_JOB) {
      /* Spark the thread pool */
      SparkThreads_LockFree(PetscPThreadRank,job_lockfree.data[PetscPThreadRank]->tcomm,job_lockfree.data[PetscPThreadRank]);
      /* Do own job */
      PetscRunKernel(PetscPThreadRank,job_lockfree.data[PetscPThreadRank]->nargs,job_lockfree.data[PetscPThreadRank]);
      /* Reset own status */ 
      job_lockfree.my_job_status[PetscPThreadRank] = THREAD_WAITING_FOR_JOB;
    }
  }

  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_PThread_LockFree"
PetscErrorCode PetscThreadCommBarrier_PThread_LockFree(PetscThreadComm tcomm)
{
  PetscInt active_threads=0,i;
  PetscBool wait=PETSC_TRUE;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;

  PetscFunctionBegin;
  /* Loop till all threads signal that they have done their job */
  while(wait) {
    for(i=0;i<tcomm->nworkThreads;i++) active_threads += job_lockfree.my_job_status[ptcomm->granks[i]];
    if(active_threads) active_threads = 0;
    else wait=PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommInitialize_LockFree"
PetscErrorCode PetscPThreadCommInitialize_LockFree(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;

  PetscFunctionBegin;

  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscThreadCommJobCtx),&job_lockfree.data);CHKERRQ(ierr);
  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&job_lockfree.my_job_status);CHKERRQ(ierr);

  /* Create threads */
  for(i=ptcomm->thread_num_start; i < tcomm->nworkThreads;i++) {
    job_lockfree.my_job_status[i] = THREAD_INITIALIZE;
    ierr = pthread_create(&ptcomm->tid[i],NULL,&PetscPThreadCommFunc_LockFree,&ptcomm->granks[i]);CHKERRQ(ierr);
  }
  if(ptcomm->ismainworker) {
    job_lockfree.my_job_status[0] = THREAD_WAITING_FOR_JOB;
  }

  /* Put a barrier so that all threads get pinned properly */
  ierr = PetscThreadCommBarrier_PThread_LockFree(tcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommFinalize_LockFree"
PetscErrorCode PetscPThreadCommFinalize_LockFree(PetscThreadComm tcomm)
{
  PetscErrorCode           ierr;
  void*                    jstatus;
  PetscThreadComm_PThread  ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscInt                 i;
  PetscFunctionBegin;
  ierr = PetscThreadCommBarrier_PThread_LockFree(tcomm);CHKERRQ(ierr);
  for(i=ptcomm->thread_num_start; i < tcomm->nworkThreads;i++) {
    job_lockfree.my_job_status[i] = THREAD_TERMINATE;
    ierr = pthread_join(ptcomm->tid[i],&jstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(job_lockfree.my_job_status);CHKERRQ(ierr);
  ierr = PetscFree(job_lockfree.data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_PThread_LockFree"
PetscErrorCode PetscThreadCommRunKernel_PThread_LockFree(MPI_Comm comm,PetscThreadCommJobCtx job)
{
  PetscErrorCode           ierr;
  PetscThreadComm          tcomm=0;
  PetscThreadComm_PThread  ptcomm;    
  
  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ptcomm = (PetscThreadComm_PThread)tcomm->data;
  if(ptcomm->nthreads) {
    PetscInt thread_num;
    /* Spark the leader thread */
    thread_num = tcomm->leader;
    /* Wait till the leader thread has finished its previous job */
    while(PetscReadOnce(int,job_lockfree.my_job_status[thread_num]) != THREAD_WAITING_FOR_JOB)
      ;
    job_lockfree.data[thread_num] = job;
    job_lockfree.my_job_status[thread_num] = THREAD_RECIEVED_JOB;
  }
  if(ptcomm->ismainworker) {
    job_lockfree.my_job_status[0] = THREAD_RECIEVED_JOB;
    job_lockfree.data[0] = job;
    PetscRunKernel(0,job->nargs, job_lockfree.data[0]);
    job_lockfree.my_job_status[0] = THREAD_WAITING_FOR_JOB;
  }
  if(ptcomm->synchronizeafter) {
    ierr = PetscThreadCommBarrier(comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
