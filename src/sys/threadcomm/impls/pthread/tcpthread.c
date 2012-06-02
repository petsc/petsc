/* Define feature test macros to make sure CPU_SET and other functions are available
 */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank;
#else
pthread_key_t PetscPThreadRankkey;
#endif

static PetscBool PetscPThreadCommInitializeCalled = PETSC_FALSE;

const char *const PetscPThreadCommSynchronizationTypes[] = {"LOCKFREE","PetscPThreadCommSynchronizationType","PTHREADSYNC_",0};
const char *const PetscPThreadCommAffinityPolicyTypes[] = {"ALL","ONECORE","NONE","PetscPThreadCommAffinityPolicyType","PTHREADAFFPOLICY_",0};
const char *const PetscPThreadCommPoolSparkTypes[] = {"LEADER","CHAIN","PetscPThreadCommPoolSparkType","PTHREADPOOLSPARK_",0};

static PetscInt ptcommcrtct = 0; /* PThread communicator creation count. Incremented whenever a pthread
                                    communicator is created and decremented when it is destroyed. On the
                                    last pthread communicator destruction, the thread pool is also terminated
                                  */

PetscInt PetscThreadCommGetRank_PThread()
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
  PetscInt                 i,icorr=0; 
  cpu_set_t                mset;
  PetscInt                 ncores,myrank=PetscThreadCommGetRank_PThread();
  PetscThreadComm          tcomm;
  PetscThreadComm_PThread  gptcomm;
  
  PetscCommGetThreadComm(PETSC_COMM_WORLD,&tcomm);
  PetscGetNCores(&ncores);
  gptcomm=(PetscThreadComm_PThread)tcomm->data;
  switch(gptcomm->aff) {
  case PTHREADAFFPOLICY_ONECORE:
    icorr = tcomm->affinities[myrank];
    CPU_ZERO(&mset);
    CPU_SET(icorr%ncores,&mset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mset);
    break;
  case PTHREADAFFPOLICY_ALL:
    CPU_ZERO(&mset);
    for(i=0;i<ncores;i++) CPU_SET(i,&mset);
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
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if(!ptcomm) PetscFunctionReturn(0);
  ptcommcrtct--;
  if(!ptcommcrtct) {
    /* Terminate the thread pool */
    ierr = (*ptcomm->finalize)(tcomm);CHKERRQ(ierr);
    ierr = PetscFree(ptcomm->tid);CHKERRQ(ierr);
  }
  ierr = PetscFree(ptcomm->granks);CHKERRQ(ierr);
  if(ptcomm->spark == PTHREADPOOLSPARK_CHAIN) {
    ierr = PetscFree(ptcomm->ngranks);CHKERRQ(ierr);
  }
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThread"
PetscErrorCode PetscThreadCommCreate_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;
  PetscInt                i;

  PetscFunctionBegin;
  ptcommcrtct++;
  ierr = PetscStrcpy(tcomm->type,PTHREAD);CHKERRQ(ierr);
  ierr = PetscNew(struct _p_PetscThreadComm_PThread,&ptcomm);CHKERRQ(ierr);
  tcomm->data = (void*)ptcomm;
  ptcomm->nthreads = 0;
  ptcomm->sync = PTHREADSYNC_LOCKFREE;
  ptcomm->aff = PTHREADAFFPOLICY_ONECORE;
  ptcomm->spark = PTHREADPOOLSPARK_LEADER;
  ptcomm->ismainworker = PETSC_TRUE;
  ptcomm->synchronizeafter = PETSC_TRUE;
  tcomm->ops->destroy = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel = PetscThreadCommRunKernel_PThread_LockFree;
  tcomm->ops->barrier   = PetscThreadCommBarrier_PThread_LockFree;
  tcomm->ops->getrank   = PetscThreadCommGetRank_PThread;

  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&ptcomm->granks);CHKERRQ(ierr);

  if(!PetscPThreadCommInitializeCalled) { /* Only done for PETSC_THREAD_COMM_WORLD */
    PetscPThreadCommInitializeCalled = PETSC_TRUE;
    PetscBool               flg1,flg2,flg3,flg4;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"PThread communicator options",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-threadcomm_pthread_main_is_worker","Main thread is also a worker thread",PETSC_NULL,PETSC_TRUE,&ptcomm->ismainworker,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-threadcomm_pthread_affpolicy","Thread affinity policy"," ",PetscPThreadCommAffinityPolicyTypes,(PetscEnum)ptcomm->aff,(PetscEnum*)&ptcomm->aff,&flg2);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-threadcomm_pthread_type","Thread pool type"," ",PetscPThreadCommSynchronizationTypes,(PetscEnum)ptcomm->sync,(PetscEnum*)&ptcomm->sync,&flg3);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-threadcomm_pthread_spark","Thread pool spark type"," ",PetscPThreadCommPoolSparkTypes,(PetscEnum)ptcomm->spark,(PetscEnum*)&ptcomm->spark,&flg4);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-threadcomm_pthread_synchronizeafter","Puts a barrier after every kernel call",PETSC_NULL,PETSC_TRUE,&ptcomm->synchronizeafter,&flg1);CHKERRQ(ierr);
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
      ptcomm->initialize      = PetscPThreadCommInitialize_LockFree;
      ptcomm->finalize        = PetscPThreadCommFinalize_LockFree;
      tcomm->ops->runkernel   = PetscThreadCommRunKernel_PThread_LockFree;
      tcomm->ops->barrier     = PetscThreadCommBarrier_PThread_LockFree;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only Lock-free synchronization scheme supported currently");
    }
    /* Set up thread ranks */
    for(i=0;i< tcomm->nworkThreads;i++) ptcomm->granks[i] = i;

    if(ptcomm->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
      PetscPThreadRank=0; /* Main thread rank */
#else
      ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
      ierr = pthread_setspecific(PetscPThreadRankkey,&ptcomm->granks[0]);CHKERRQ(ierr);
#endif
    }
    /* Set the leader thread rank */
    if(ptcomm->nthreads) {
      if(ptcomm->ismainworker) tcomm->leader = ptcomm->granks[1];
      else tcomm->leader = ptcomm->granks[0];
    }
  
    if(ptcomm->spark == PTHREADPOOLSPARK_CHAIN) {
      ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&ptcomm->ngranks);CHKERRQ(ierr);
      for(i=ptcomm->thread_num_start;i < tcomm->nworkThreads-1;i++) ptcomm->ngranks[i] = ptcomm->granks[i+1];
      ptcomm->ngranks[tcomm->nworkThreads-1] = -1;
    }

    /* Create array holding pthread ids */
    ierr = PetscMalloc(tcomm->nworkThreads*sizeof(pthread_t),&ptcomm->tid);CHKERRQ(ierr);

    /* Set affinity of the main thread */
    
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
    cpu_set_t mset;
    PetscInt  ncores,icorr;
    

    ierr = PetscGetNCores(&ncores);CHKERRQ(ierr);
    CPU_ZERO(&mset);
    icorr = tcomm->affinities[0]%ncores;
    CPU_SET(icorr,&mset);
    sched_setaffinity(0,sizeof(cpu_set_t),&mset);
#endif

    /* Initialize thread pool */
    ierr = (*ptcomm->initialize)(tcomm);CHKERRQ(ierr);

  } else {
    PetscThreadComm          gtcomm;
    PetscThreadComm_PThread  gptcomm;
    PetscInt                 *granks,j,*gaffinities;

    ierr = PetscCommGetThreadComm(PETSC_COMM_WORLD,&gtcomm);CHKERRQ(ierr);
    gaffinities = gtcomm->affinities;
    gptcomm = (PetscThreadComm_PThread)tcomm->data;
    granks = gptcomm->granks;
    /* Copy over the data from the global thread communicator structure */
    ptcomm->ismainworker     = gptcomm->ismainworker;
    ptcomm->thread_num_start = gptcomm->thread_num_start;
    ptcomm->sync             = gptcomm->sync;
    ptcomm->aff              = gptcomm->aff;
    tcomm->ops->runkernel    = gtcomm->ops->runkernel;
    tcomm->ops->barrier      = gtcomm->ops->barrier;
    
    for(i=0; i < tcomm->nworkThreads;i++) {
      for(j=0;j < gtcomm->nworkThreads; j++) {
	if(tcomm->affinities[i] == gaffinities[j]) {
	  ptcomm->granks[i] = granks[j];
	}
      }
    }
  }  

  PetscFunctionReturn(0);
}
EXTERN_C_END
    
