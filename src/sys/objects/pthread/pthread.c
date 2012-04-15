#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

/* Initialize global variables and function pointers */
PetscBool   PetscThreadGo = PETSC_TRUE;
PetscMPIInt PetscMaxThreads = -1;
pthread_t*  PetscThreadPoint=NULL;
PetscInt*   PetscThreadsCoreAffinities=NULL;
PetscInt    PetscMainThreadShareWork = 1;
PetscInt    PetscMainThreadCoreAffinity = 0;
PetscBool   PetscThreadsInitializeCalled = PETSC_FALSE;
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscThreadRank;
#else
pthread_key_t PetscThreadsRankkey;
#endif

PetscInt*   PetscThreadRanks;

void*          (*PetscThreadFunc)(void*) = NULL;
PetscErrorCode (*PetscThreadsSynchronizationInitialize)(PetscInt) = NULL;
PetscErrorCode (*PetscThreadsSynchronizationFinalize)(void) = NULL;
void*          (*PetscThreadsWait)(void*) = NULL;
PetscErrorCode (*PetscThreadsRunKernel)(PetscErrorCode (*pFunc)(void*),void**,PetscInt,PetscInt*)=NULL;

static const char *const PetscThreadsSynchronizationTypes[] = {"NOPOOL","MAINPOOL","TRUEPOOL","CHAINPOOL","TREEPOOL","LOCKFREE","PetscThreadsSynchronizationType","THREADSYNC_",0};
static const char *const PetscThreadsAffinityPolicyTypes[] = {"ALL","ONECORE","NONE","ThreadAffinityPolicyType","THREADAFFINITYPOLICY_",0};

static PetscThreadsAffinityPolicyType thread_aff_policy=THREADAFFINITYPOLICY_ONECORE;

static PetscInt     N_CORES;

PetscErrorCode PetscThreadsFinish(void* arg) {
  PetscThreadGo = PETSC_FALSE;
  return(0);
}

PETSC_STATIC_INLINE PetscInt PetscGetThreadRank()
{
#if defined(PETSC_PTHREAD_LOCAL)
  return PetscThreadRank;
#else
  return *((PetscInt*)pthread_getspecific(PetscThreadsRankkey));
#endif
}

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
/* Set CPU affinity for the main thread, only called by main thread */
void PetscSetMainThreadAffinity(PetscInt icorr)
{
  cpu_set_t mset;

  CPU_ZERO(&mset);
  CPU_SET(icorr%N_CORES,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);
}

/* Only called by spawned threads */
void PetscThreadsDoCoreAffinity(void)
{
  PetscInt  i,icorr=0; 
  cpu_set_t mset;
  PetscInt  myrank=PetscGetThreadRank();
  
  switch(thread_aff_policy) {
  case THREADAFFINITYPOLICY_ONECORE:
    icorr = PetscThreadsCoreAffinities[myrank];
    CPU_ZERO(&mset);
    CPU_SET(icorr%N_CORES,&mset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mset);
    break;
  case THREADAFFINITYPOLICY_ALL:
    CPU_ZERO(&mset);
    for(i=0;i<N_CORES;i++) CPU_SET(i,&mset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mset);
    break;
  case THREADAFFINITYPOLICY_NONE:
    break;
  }
}
#endif

/* Sets the CPU affinities for threads */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadsSetAffinities"
PetscErrorCode PetscThreadsSetAffinities(PetscInt affinities[])
{
  PetscErrorCode ierr;
  PetscInt       nworkThreads=PetscMaxThreads+PetscMainThreadShareWork;
  PetscInt       nmax=nworkThreads;
  PetscBool      flg;

  PetscFunctionBegin;

  ierr = PetscMalloc(nworkThreads*sizeof(PetscInt),&PetscThreadsCoreAffinities);CHKERRQ(ierr);

  if(affinities == PETSC_NULL) {
    /* PETSc decides affinities */
    /* Check if the run-time option is set */
    ierr = PetscOptionsIntArray("-thread_affinities","Set CPU affinities of threads","PetscThreadsSetAffinities",PetscThreadsCoreAffinities,&nmax,&flg);CHKERRQ(ierr);
    if(flg) {
      if(nmax != nworkThreads) {
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, Threads = %D, CPU affinities set = %D",nworkThreads,nmax);
      }
    } else {
      /* PETSc default affinities */
      PetscInt i;
      if(PetscMainThreadShareWork) {
	PetscThreadsCoreAffinities[0] = PetscMainThreadCoreAffinity;
	for(i=1; i< nworkThreads; i++) PetscThreadsCoreAffinities[i] = i%N_CORES;
      } else {
	for(i=0;i < nworkThreads;i++) PetscThreadsCoreAffinities[i] = (i+1)%N_CORES;
      }
    }
  } else {
    /* Set user provided affinities */
    ierr = PetscMemcpy(PetscThreadsCoreAffinities,affinities,PetscMaxThreads*sizeof(PetscInt));
  }
    PetscFunctionReturn(0);
  }
      
#undef __FUNCT__
#define __FUNCT__ "PetscThreadsInitialize"
/*
  PetscThreadsInitialize - Initializes the thread synchronization scheme with given
  of threads.

  Input Parameters:
. nthreads - Number of threads to create

  Level: beginner

.seealso: PetscThreadsFinalize()
*/
PetscErrorCode PetscThreadsInitialize(PetscInt nthreads)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       nworkThreads=PetscMaxThreads+PetscMainThreadShareWork;

  PetscFunctionBegin;
  if(PetscThreadsInitializeCalled) PetscFunctionReturn(0);

  /* Set thread ranks */
  ierr = PetscMalloc(nworkThreads*sizeof(PetscInt),&PetscThreadRanks);CHKERRQ(ierr);
  for(i=0;i< nworkThreads;i++) PetscThreadRanks[i] = i;
#if defined(PETSC_PTHREAD_LOCAL)
  if(PetscMainThreadShareWork) PetscThreadRank=0; /* Main thread rank */
#else
  ierr = pthread_key_create(&PetscThreadsRankkey,NULL);CHKERRQ(ierr);
  if(PetscMainThreadShareWork) {
    ierr = pthread_setspecific(PetscThreadsRankkey,&PetscThreadRanks[0]);CHKERRQ(ierr);
  }
#endif
  /* Set thread affinities */
  ierr = PetscThreadsSetAffinities(PETSC_NULL);CHKERRQ(ierr);
  /* Initialize thread pool */
  if(PetscThreadsSynchronizationInitialize) {
    ierr = (*PetscThreadsSynchronizationInitialize)(nthreads);CHKERRQ(ierr);
  }
  PetscThreadsInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsFinalize"
/*
  PetscThreadsFinalize - Terminates the thread synchronization scheme initiated
  by PetscThreadsInitialize()

  Level: beginner

.seealso: PetscThreadsInitialize()
*/
PetscErrorCode PetscThreadsFinalize(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if(!PetscThreadsInitializeCalled) PetscFunctionReturn(0);

  if (PetscThreadsSynchronizationFinalize) {
    ierr = (*PetscThreadsSynchronizationFinalize)();CHKERRQ(ierr);
  }

  ierr = PetscFree(PetscThreadsCoreAffinities);CHKERRQ(ierr);
  ierr = PetscFree(PetscThreadRanks);CHKERRQ(ierr);
  PetscThreadsInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSetMaxPThreads"
/*
   PetscSetMaxPThreads - Sets the number of pthreads to be used.

   Not collective
  
   Input Parameters:
.  nthreads - # of pthreads.

   Options Database Keys:
   -nthreads <nthreads> Number of pthreads to be used.

   Level: developer
 
   Notes:
   Use nthreads = PETSC_DECIDE for PETSc to calculate the maximum number of pthreads to be used.
   If nthreads = PETSC_DECIDE, PETSc will create (ncpus - 1) threads where ncpus is the number of 
   available processing cores. 
   
   By default, the main execution thread is also considered as a work thread.
   
   
.seealso: PetscGetMaxPThreads()
*/ 
PetscErrorCode PetscSetMaxPThreads(PetscInt nthreads) 
{
  PetscErrorCode ierr;
  PetscBool      flg=PETSC_FALSE;
  PetscInt       nworkThreads;

  PetscFunctionBegin;

  N_CORES=1; /* Default value if N_CORES cannot be found out */
  /* Find the number of cores */
#if defined(PETSC_HAVE_SCHED_CPU_SET_T) /* Linux */
  N_CORES = get_nprocs();
#elif defined(PETSC_HAVE_SYS_SYSCTL_H) /* MacOS, BSD */
  {
    size_t   len = sizeof(N_CORES);
    ierr = sysctlbyname("hw.activecpu",&N_CORES,&len,NULL,0);CHKERRQ(ierr);
  }
#elif defined(PETSC_HAVE_WINDOWS_H)   /* Windows */
  {
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    N_CORES = sysinfo.dwNumberOfProcessors;
  }
#endif
  PetscMaxThreads=N_CORES-1;
  if(nthreads == PETSC_DECIDE) {
    /* Check if run-time option is given */
    ierr = PetscOptionsInt("-nthreads","Set number of threads to be used for the thread pool","PetscSetMaxPThreads",N_CORES,&nworkThreads,&flg);CHKERRQ(ierr);
    if(flg) PetscMaxThreads = nworkThreads-1;
  } else PetscMaxThreads = nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGetMaxPThreads"
/*
   PetscGetMaxPThreads - Returns the number of pthreads used in the thread pool.

   Not collective
  
   Output Parameters:
.  nthreads - Number of pthreads in the the thread pool.

   Level: beginner
 
   Notes:
   Must call PetscSetMaxPThreads() before
   
.seealso: PetscSetMaxPThreads()
*/ 
PetscErrorCode PetscGetMaxPThreads(PetscInt *nthreads)
{
  PetscFunctionBegin;
  if(PetscMaxThreads < 0) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscSetMaxPThreads() first");
  } else {
    *nthreads = PetscMaxThreads+PetscMainThreadShareWork;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsCheckInitial_Private_Pthread"
PetscErrorCode PetscOptionsCheckInitial_Private_Pthread(void)
{
  PetscErrorCode                  ierr;
  PetscBool                       flg1=PETSC_FALSE;
  PetscThreadsSynchronizationType thread_sync_type=THREADSYNC_LOCKFREE;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"PThread Options","Sys");CHKERRQ(ierr);

  /* Set nthreads */
  ierr = PetscSetMaxPThreads(PETSC_DECIDE);CHKERRQ(ierr);

  /* Check to see if the user wants the main thread not to share work with the other threads */
  ierr = PetscOptionsInt("-mainthread_is_worker","Main thread is also a work thread",PETSC_NULL,PetscMainThreadShareWork,&PetscMainThreadShareWork,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mainthread_affinity","CPU affinity of main thread","PetscSetMainThreadAffinity",PetscMainThreadCoreAffinity,&PetscMainThreadCoreAffinity,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  PetscSetMainThreadAffinity(PetscMainThreadCoreAffinity);
#endif
 
  /* Get thread affinity policy */
  ierr = PetscOptionsEnum("-thread_aff_policy","Type of thread affinity policy"," ",PetscThreadsAffinityPolicyTypes,(PetscEnum)thread_aff_policy,(PetscEnum*)&thread_aff_policy,&flg1);CHKERRQ(ierr);
  /* Get thread synchronization scheme */
  ierr = PetscOptionsEnum("-thread_sync_type","Type of thread synchronization algorithm"," ",PetscThreadsSynchronizationTypes,(PetscEnum)thread_sync_type,(PetscEnum*)&thread_sync_type,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
  switch(thread_sync_type) {
#if 0 /* I'm tired of modifying each thread pool whenever there is a common change in any one. Hence, i'm disabling
         all the thread pools except lockfree for now. Will activate them once all the other development work
         is done.
      */
    case THREADSYNC_TREEPOOL:
      PetscThreadFunc       = &PetscThreadFunc_Tree;
      PetscThreadsSynchronizationInitialize = &PetscThreadsSynchronizationInitialize_Tree;
      PetscThreadsSynchronizationFinalize   = &PetscThreadsSynchronizationFinalize_Tree;
      PetscThreadsWait      = &PetscThreadsWait_Tree;
      PetscThreadsRunKernel = &PetscThreadsRunKernel_Tree;
      PetscInfo1(PETSC_NULL,"Using tree thread pool with %d threads\n",PetscMaxThreads);
      break;
    case THREADSYNC_MAINPOOL:
      PetscThreadFunc       = &PetscThreadFunc_Main;
      PetscThreadsSynchronizationInitialize = &PetscThreadsSynchronizationInitialize_Main;
      PetscThreadsSynchronizationFinalize   = &PetscThreadsSynchronizationFinalize_Main;
    PetscThreadsWait      = &PetscThreadsWait_Main;
    PetscThreadsRunKernel = &PetscThreadsRunKernel_Main;
    PetscInfo1(PETSC_NULL,"Using main thread pool with %d threads\n",PetscMaxThreads);
    break;
    case THREADSYNC_CHAINPOOL:
      PetscThreadFunc       = &PetscThreadFunc_Chain;
      PetscThreadsSynchronizationInitialize = &PetscThreadsSynchronizationInitialize_Chain;
      PetscThreadsSynchronizationFinalize   = &PetscThreadsSynchronizationFinalize_Chain;
      PetscThreadsWait      = &PetscThreadsWait_Chain;
      PetscThreadsRunKernel = &PetscThreadsRunKernel_Chain;
      PetscInfo1(PETSC_NULL,"Using chain thread pool with %d threads\n",PetscMaxThreads);
      break;
    case THREADSYNC_TRUEPOOL:
#if defined(PETSC_HAVE_PTHREAD_BARRIER_T)
      PetscThreadFunc       = &PetscThreadFunc_True;
      PetscThreadsSynchronizationInitialize = &PetscThreadsSynchronizationInitialize_True;
      PetscThreadsSynchronizationFinalize   = &PetscThreadsSynchronizationFinalize_True;
      PetscThreadsWait      = &PetscThreadsWait_True;
      PetscThreadsRunKernel = &PetscThreadsRunKernel_True;
      PetscInfo1(PETSC_NULL,"Using true thread pool with %d threads\n",PetscMaxThreads);
#else
      PetscThreadFunc       = &PetscThreadFunc_Main;
      PetscThreadsSynchronizationInitialize = &PetscThreadsSynchronizationInitialize_Main;
      PetscThreadsSynchronizationFinalize   = &PetscThreadsSynchronizationFinalize_Main;
      PetscThreadsWait      = &PetscThreadsWait_Main;
      PetscThreadsRunKernel = &PetscThreadsRunKernel_Main;
      PetscInfo1(PETSC_NULL,"Cannot use true thread pool since pthread_barrier_t is not defined, creating main thread pool instead with %d threads\n",PetscMaxThreads);
#endif
      break;
    case THREADSYNC_NOPOOL:
      PetscThreadsSynchronizationInitialize = PETSC_NULL;
      PetscThreadsSynchronizationFinalize   = PETSC_NULL;
      PetscThreadFunc       = &PetscThreadFunc_None;
      PetscThreadsWait      = &PetscThreadsWait_None;
      PetscThreadsRunKernel = &PetscThreadsRunKernel_None;
      PetscInfo1(PETSC_NULL,"Using No thread pool with %d threads\n",PetscMaxThreads);
      break;
#endif
  case THREADSYNC_LOCKFREE:
    PetscThreadFunc       = &PetscThreadFunc_LockFree;
    PetscThreadsSynchronizationInitialize = &PetscThreadsSynchronizationInitialize_LockFree;
    PetscThreadsSynchronizationFinalize   = &PetscThreadsSynchronizationFinalize_LockFree;
    PetscThreadsWait      = &PetscThreadsWait_LockFree;
    PetscThreadsRunKernel = &PetscThreadsRunKernel_LockFree;
    PetscInfo1(PETSC_NULL,"Using lock-free thread synchronization with %d threads\n",PetscMaxThreads+PetscMainThreadShareWork);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only Lock-free synchronization scheme supported currently");
  }
  PetscFunctionReturn(0);
}
