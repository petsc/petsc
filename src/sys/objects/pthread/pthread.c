#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>

/* Initialize global variables and function pointers */
PetscBool   PetscThreadGo = PETSC_TRUE;
PetscMPIInt PetscMaxThreads = -1;
pthread_t*  PetscThreadPoint=NULL;
PetscInt*   ThreadCoreAffinity=NULL;
PetscInt    PetscMainThreadShareWork = 1;
PetscInt    MainThreadCoreAffinity = 0;
PetscBool   PetscThreadsInitializeCalled = PETSC_FALSE;
pthread_key_t rankkey;
PetscInt*   threadranks=NULL;

void*          (*PetscThreadFunc)(void*) = NULL;
PetscErrorCode (*PetscThreadsSynchronizationInitialize)(PetscInt) = NULL;
PetscErrorCode (*PetscThreadsSynchronizationFinalize)(void) = NULL;
void*          (*PetscThreadsWait)(void*) = NULL;
PetscErrorCode (*PetscThreadsRunKernel)(void* (*pFunc)(void*),void**,PetscInt,PetscInt*)=NULL;

const char *const ThreadSynchronizationTypes[] = {"NOPOOL","MAINPOOL","TRUEPOOL","CHAINPOOL","TREEPOOL","LOCKFREE","ThreadSynchronizationType","THREADSYNC_",0};
const char *const ThreadAffinityPolicyTypes[] = {"ALL","ONECORE","NONE","ThreadAffinityPolicyType","THREADAFFINITYPOLICY_",0};

static ThreadAffinityPolicyType thread_aff_policy=THREADAFFINITYPOLICY_ONECORE;

static PetscInt     N_CORES;

void* FuncFinish(void* arg) {
  PetscThreadGo = PETSC_FALSE;
  return(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscGetThreadRank"
/*
  PetscGetThreadRank - Gets the rank of the calling thread.

  Level: developer

  Notes: The ranks of all the threads spawned via PetscThreadsInitialize() start from 1 and the
         main control thread assigned rank 0.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscGetThreadRank(PetscInt* rankp)
{
  PetscInt trank;

  PetscFunctionBegin;
  trank = *(PetscInt*)pthread_getspecific(rankkey);
  *rankp = trank;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
/* Set CPU affinity for the main thread */
void PetscSetMainThreadAffinity(PetscInt icorr)
{
  cpu_set_t mset;

  CPU_ZERO(&mset);
  CPU_SET(icorr%N_CORES,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);
}

void DoCoreAffinity(void)
{
  PetscInt  i,icorr=0; 
  pthread_t pThread = pthread_self();
  cpu_set_t mset;
  PetscInt  myrank;
  
  PetscGetThreadRank(&myrank);

  switch(thread_aff_policy) {
  case THREADAFFINITYPOLICY_ONECORE:
    for (i=0; i<PetscMaxThreads; i++) {
      if (pthread_equal(pThread,PetscThreadPoint[i])) {
	icorr = ThreadCoreAffinity[i];
	CPU_ZERO(&mset);
	CPU_SET(icorr%N_CORES,&mset);
	pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&mset);
	break;
      }
    }
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
  char           tstr[9];
  char           tbuf[2];
  PetscInt       i;
  PetscBool      flg1;

  PetscFunctionBegin;
  if(PetscThreadsInitializeCalled) PetscFunctionReturn(0);

  /* Set default affinities for threads: each thread has an affinity to one core unless the PetscMaxThreads > N_CORES */
  ierr = PetscMalloc(PetscMaxThreads*sizeof(PetscInt),&ThreadCoreAffinity);CHKERRQ(ierr);
  
  strcpy(tstr,"-thread");
  for(i=0;i<PetscMaxThreads;i++) {
    ThreadCoreAffinity[i] = i+PetscMainThreadShareWork;
    sprintf(tbuf,"%d",i);
    strcat(tstr,tbuf);
    ierr = PetscOptionsHasName(PETSC_NULL,tstr,&flg1);CHKERRQ(ierr);
    if(flg1) {
      ierr = PetscOptionsGetInt(PETSC_NULL,tstr,&ThreadCoreAffinity[i],PETSC_NULL);CHKERRQ(ierr);
      ThreadCoreAffinity[i] = ThreadCoreAffinity[i]%N_CORES; /* check on the user */
    }
    tstr[7] = '\0';
  }

  /* Create key to store the thread rank */
  ierr = pthread_key_create(&rankkey,NULL);CHKERRQ(ierr);
  /* Create array to store thread ranks */
  ierr = PetscMalloc((PetscMaxThreads+1)*sizeof(PetscInt),&threadranks);CHKERRQ(ierr);
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

  ierr = pthread_key_delete(rankkey);CHKERRQ(ierr);
  ierr = PetscFree(ThreadCoreAffinity);CHKERRQ(ierr);
  ierr = PetscFree(threadranks);CHKERRQ(ierr);
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
    size_t   len = sizeof(N_CORES);
    ierr = sysctlbyname("hw.activecpu",&N_CORES,&len,NULL,0);CHKERRQ(ierr);
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
  ThreadSynchronizationType      thread_sync_type=THREADSYNC_LOCKFREE;

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
    ierr = PetscOptionsGetInt(PETSC_NULL,"-main",&MainThreadCoreAffinity,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
    PetscSetMainThreadAffinity(MainThreadCoreAffinity);
#endif
  }
 
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"PThread Options","Sys");CHKERRQ(ierr);
  /* Get thread affinity policy */
  ierr = PetscOptionsEnum("-thread_aff_policy","Type of thread affinity policy"," ",ThreadAffinityPolicyTypes,(PetscEnum)thread_aff_policy,(PetscEnum*)&thread_aff_policy,&flg1);CHKERRQ(ierr);
  /* Get thread synchronization scheme */
  ierr = PetscOptionsEnum("-thread_sync_type","Type of thread synchronization algorithm"," ",ThreadSynchronizationTypes,(PetscEnum)thread_sync_type,(PetscEnum*)&thread_sync_type,&flg1);CHKERRQ(ierr);
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
