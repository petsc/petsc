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
PETSC_PTHREAD_LOCAL PetscInt PetscThreadRank;

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

#define PetscGetThreadRank() (PetscThreadRank)
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
/* Set CPU affinity for the main thread */
void PetscSetMainThreadAffinity(PetscInt icorr)
{
  cpu_set_t mset;

  CPU_ZERO(&mset);
  CPU_SET(icorr%N_CORES,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);
}

void PetscThreadsDoCoreAffinity(void)
{
  PetscInt  i,icorr=0; 
  cpu_set_t mset;
  PetscInt  myrank=PetscGetThreadRank();
  
  switch(thread_aff_policy) {
  case THREADAFFINITYPOLICY_ONECORE:
    if(myrank == 0) icorr = PetscMainThreadCoreAffinity;
    else icorr = PetscThreadsCoreAffinities[myrank-1];
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
  PetscInt       nmax=PetscMaxThreads;
  PetscBool      flg;

  PetscFunctionBegin;

  ierr = PetscMalloc(PetscMaxThreads*sizeof(PetscInt),&PetscThreadsCoreAffinities);CHKERRQ(ierr);

  if(affinities == PETSC_NULL) {
    /* PETSc decides affinities */
    /* Check if the run-time option is set */
    ierr = PetscOptionsIntArray("-thread_affinities","Set CPU affinities of threads","PetscThreadsSetAffinities",PetscThreadsCoreAffinities,&nmax,&flg);CHKERRQ(ierr);
    if(flg) {
      if(nmax != PetscMaxThreads) {
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, Threads = %D, CPU affinities set = %D",PetscMaxThreads,nmax);
      }
    } else {
      /* PETSc default affinities */
      PetscInt i;
      for(i=0; i< PetscMaxThreads; i++) PetscThreadsCoreAffinities[i] = (i+PetscMainThreadShareWork)%N_CORES;
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


  PetscFunctionBegin;
  if(PetscThreadsInitializeCalled) PetscFunctionReturn(0);

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
  PetscThreadsInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSetMaxPThreads"
/*
   PetscSetMaxPThreads - Sets the number of pthreads to create.

   Not collective
  
   Input Parameters:
.  nthreads - # of pthreads.

   Options Database Keys:
   -nthreads <nthreads> Number of pthreads to create.

   Level: beginner
 
   Notes:
   Use nthreads = PETSC_DECIDE for PETSc to calculate the maximum number of pthreads to create.
   By default, the main execution thread is also considered as a work thread. Hence, PETSc will 
   create (ncpus - 1) threads where ncpus is the number of processing cores available. 
   The option -mainthread_no_share_work can be used to have the main thread act as a controller only. 
   For this case, PETSc will create ncpus threads.
   
.seealso: PetscGetMaxPThreads()
*/ 
PetscErrorCode PetscSetMaxPThreads(PetscInt nthreads) 
{
  PetscErrorCode ierr;
  PetscBool      flg=PETSC_FALSE;

  PetscFunctionBegin;

  N_CORES=1; /* Default value if N_CORES cannot be found out */
  PetscMaxThreads = N_CORES;
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

  if(nthreads == PETSC_DECIDE) {
    /* Check if run-time option is given */
    ierr = PetscOptionsGetInt(PETSC_NULL,"-nthreads",&PetscMaxThreads,&flg);CHKERRQ(ierr);
    if(!flg) {
      PetscMaxThreads = N_CORES - PetscMainThreadShareWork;
    } 
  } else PetscMaxThreads = nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGetMaxPThreads"
/*
   PetscGetMaxPThreads - Returns the number of pthreads created.

   Not collective
  
   Output Parameters:
.  nthreads - Number of pthreads created.

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
    *nthreads = PetscMaxThreads;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsCheckInitial_Private_Pthread"
PetscErrorCode PetscOptionsCheckInitial_Private_Pthread(void)
{
  PetscErrorCode                 ierr;
  PetscBool                      flg1=PETSC_FALSE;
  PetscThreadsSynchronizationType      thread_sync_type=THREADSYNC_LOCKFREE;

  PetscFunctionBegin;

  /* Check to see if the user wants the main thread not to share work with the other threads */
  ierr = PetscOptionsHasName(PETSC_NULL,"-mainthread_no_share_work",&flg1);CHKERRQ(ierr);
  if(flg1) PetscMainThreadShareWork = 0;

  /*
      Set maximum number of threads
  */
  ierr = PetscSetMaxPThreads(PETSC_DECIDE);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-main",&flg1);CHKERRQ(ierr);
  if(flg1) {
    ierr = PetscOptionsGetInt(PETSC_NULL,"-main",&PetscMainThreadCoreAffinity,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
    PetscSetMainThreadAffinity(PetscMainThreadCoreAffinity);
#endif
  }
 
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"PThread Options","Sys");CHKERRQ(ierr);
  /* Get thread affinity policy */
  ierr = PetscOptionsEnum("-thread_aff_policy","Type of thread affinity policy"," ",PetscThreadsAffinityPolicyTypes,(PetscEnum)thread_aff_policy,(PetscEnum*)&thread_aff_policy,&flg1);CHKERRQ(ierr);
  /* Get thread synchronization scheme */
  ierr = PetscOptionsEnum("-thread_sync_type","Type of thread synchronization algorithm"," ",PetscThreadsSynchronizationTypes,(PetscEnum)thread_sync_type,(PetscEnum*)&thread_sync_type,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
  switch(thread_sync_type) {
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
  case THREADSYNC_LOCKFREE:
    PetscThreadFunc       = &PetscThreadFunc_LockFree;
    PetscThreadsSynchronizationInitialize = &PetscThreadsSynchronizationInitialize_LockFree;
    PetscThreadsSynchronizationFinalize   = &PetscThreadsSynchronizationFinalize_LockFree;
    PetscThreadsWait      = &PetscThreadsWait_LockFree;
    PetscThreadsRunKernel = &PetscThreadsRunKernel_LockFree;
    PetscInfo1(PETSC_NULL,"Using lock-free thread synchronization with %d threads\n",PetscMaxThreads);
    break;
  }
  PetscFunctionReturn(0);
}
