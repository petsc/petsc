#include <../src/sys/threadcomm/impls/pthread/pthreadimpl.h>

#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank;
#else
pthread_key_t PetscPThreadRankkey;
#endif

static PetscBool PetscPThreadCommInitializeCalled = PETSC_FALSE;

const char *const PetscPThreadCommSynchronizationTypes[] = {"LOCKFREE","PetscPThreadCommSynchronizationType","PTHREADSYNC_",0};
const char *const PetscPThreadCommAffinityPolicyTypes[] = {"ALL","ONECORE","NONE","PetscPThreadCommAffinityPolicyType","PTHREADAFFPOLICY_",0};

static PetscInt ptcommcrtct = 0; /* PThread communicator creation count. Incremented whenever a pthread
                                    communicator is created and decremented when it is destroyed. On the
                                    last pthread communicator destruction, the thread pool is also terminated
                                  */

PetscInt PetscGetPThreadRank()
{
#if defined(PETSC_PTHREAD_LOCAL)
  return PetscPThreadRank;
#else
  return *((PetscInt*)pthread_getspecific(PetscPThreadRankkey));
#endif
}


#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
void PetscPThreadCommDoCoreAffinity(void)
{
  PetscInt  i,icorr=0; 
  cpu_set_t mset;
  PetscInt  myrank=PetscGetPThreadRank();
  PetscThreadComm_PThread *gptcomm=(PetscThreadComm_PThread*)PETSC_THREAD_COMM_WORLD->data;
  
  switch(gptcomm->aff) {
  case PTHREADAFFPOLICY_ONECORE:
    icorr = PETSC_THREAD_COMM_WORLD->affinities[myrank];
    CPU_ZERO(&mset);
    CPU_SET(icorr%N_CORES,&mset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mset);
    break;
  case PTHREADAFFPOLICY_ALL:
    CPU_ZERO(&mset);
    for(i=0;i<N_CORES;i++) CPU_SET(i,&mset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mset);
    break;
  case PTHREADAFFPOLICY_NONE:
    break;
  }
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy_PThread"
PetscErrorCode PetscThreadCommDestroy_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread *ptcomm=(PetscThreadComm_PThread*)tcomm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(!ptcomm) PetscFunctionReturn(0);
  ptcommcrtct--;
  if(!ptcommcrtct) {
    /* Terminate the thread pool */
    ierr = (*ptcomm->finalize)(tcomm);CHKERRQ(ierr);
    ierr = PetscFree(ptcomm->tid);CHKERRQ(ierr);
  }
  ierr = PetscFree(ptcomm->ranks);CHKERRQ(ierr);
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_PThread"
PetscErrorCode PetscThreadCommRunKernel_PThread(PetscThreadComm tcomm,PetscErrorCode (*pFunc)(void*),void** pdata)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread *ptcomm=(PetscThreadComm_PThread*)tcomm->data;

  PetscFunctionBegin;
  ierr = (*ptcomm->runkernel)(tcomm,pFunc,pdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThread"
PetscErrorCode PetscThreadCommCreate_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread *ptcomm;
  PetscErrorCode          ierr;
  PetscInt                i;

  PetscFunctionBegin;
  ptcommcrtct++;
  ierr = PetscNewLog(tcomm,PetscThreadComm_PThread,&ptcomm);CHKERRQ(ierr);
  tcomm->data = (void*)ptcomm;
  ptcomm->nthreads = 0;
  ptcomm->sync = PTHREADSYNC_LOCKFREE;
  ptcomm->aff = PTHREADAFFPOLICY_ONECORE;
  ptcomm->ismainworker = PETSC_TRUE;
  tcomm->ops->destroy = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel = PetscThreadCommRunKernel_PThread;

  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&ptcomm->ranks);CHKERRQ(ierr);

  if(!PetscPThreadCommInitializeCalled) { /* Only done for PETSC_THREAD_COMM_WORLD */
    PetscPThreadCommInitializeCalled = PETSC_TRUE;
    PetscBool               flg1,flg2,flg3;

    ierr = PetscObjectOptionsBegin((PetscObject)tcomm);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-threadcomm_pthread_main_is_worker","Main thread is also a worker thread",PETSC_NULL,PETSC_TRUE,&ptcomm->ismainworker,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-threadcomm_pthread_affpolicy","Thread affinity policy"," ",PetscPThreadCommAffinityPolicyTypes,(PetscEnum)ptcomm->aff,(PetscEnum*)&ptcomm->aff,&flg2);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-threadcomm_pthread_type","Thread pool type"," ",PetscPThreadCommSynchronizationTypes,(PetscEnum)ptcomm->sync,(PetscEnum*)&ptcomm->sync,&flg3);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    if(ptcomm->ismainworker) {
      ptcomm->nthreads = tcomm->nworkThreads-1;
      ptcomm->thread_num_start = 1;
    } else {
      ptcomm->nthreads = tcomm->nworkThreads;
      ptcomm->thread_num_start = 0;
    }

    switch(ptcomm->sync) {
    case PTHREADSYNC_LOCKFREE:
      ptcomm->initialize = PetscPThreadCommInitialize_LockFree;
      ptcomm->finalize   = PetscPThreadCommFinalize_LockFree;
      ptcomm->runkernel = PetscPThreadCommRunKernel_LockFree;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only Lock-free synchronization scheme supported currently");
    }
    /* Set up thread ranks */
    for(i=0;i< tcomm->nworkThreads;i++) ptcomm->ranks[i] = i;

    if(ptcomm->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
      PetscPThreadRank=0; /* Main thread rank */
#else
      ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
      ierr = pthread_setspecific(PetscPThreadRankkey,&ptcomm->ranks[0]);CHKERRQ(ierr);
#endif
    }
    /* Create array holding pthread ids */
    ierr = PetscMalloc(tcomm->nworkThreads*sizeof(pthread_t),&ptcomm->tid);CHKERRQ(ierr);

    /* Initialize thread pool */
    ierr = (*ptcomm->initialize)(tcomm);CHKERRQ(ierr);

  } else {
    PetscThreadComm_PThread *gptcomm=(PetscThreadComm_PThread*)PETSC_THREAD_COMM_WORLD->data;
    PetscInt *granks=gptcomm->ranks;
    PetscInt j;
    PetscInt *gaffinities=PETSC_THREAD_COMM_WORLD->affinities;

    /* Copy over the data from the global PETSC_THREAD_COMM_WORLD structure */
    ptcomm->ismainworker     = gptcomm->ismainworker;
    ptcomm->thread_num_start = gptcomm->thread_num_start;
    ptcomm->sync             = gptcomm->sync;
    ptcomm->aff              = gptcomm->aff;
    ptcomm->runkernel        = gptcomm->runkernel;
    
    for(i=0; i < tcomm->nworkThreads;i++) {
      for(j=0;j < PETSC_THREAD_COMM_WORLD->nworkThreads; j++) {
	if(tcomm->affinities[i] == gaffinities[j]) {
	  ptcomm->ranks[i] = granks[j];
	}
      }
    }
  }  
  PetscFunctionReturn(0);
}
EXTERN_C_END
    
