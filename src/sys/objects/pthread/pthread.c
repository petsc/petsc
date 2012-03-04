#include <petscsys.h>        /*I  "petscsys.h"   I*/
#include <../src/sys/objects/pthread/pthreadimpl.h>
#include <private/vecimpl.h>

/* Initialize global variables and function pointers */
PetscBool   PetscThreadGo = PETSC_TRUE;
PetscMPIInt PetscMaxThreads = -1;
pthread_t*  PetscThreadPoint=NULL;
PetscInt*   ThreadCoreAffinity=NULL;
PetscInt    PetscMainThreadShareWork = 1;
PetscInt    MainThreadCoreAffinity = 0;
PetscBool   PetscThreadsInitializeCalled = PETSC_FALSE;

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
  int       i,icorr=0; 
  pthread_t pThread = pthread_self();
  cpu_set_t mset;
  
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
  ierr = PetscFree(ThreadCoreAffinity);CHKERRQ(ierr);
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
   The number of threads is then set to the number of processing units available
   for the system. By default, PETSc will set max. threads = # of processing units
   available - 1 (since we consider the main thread as also a worker thread). If the
   option -mainthread_no_share_work is used, then max. threads created = # of
   available processing units.
   
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
#define __FUNCT__ "PetscThreadsLayoutCreate"
/*
     PetscThreadsLayoutCreate - Allocates PetsThreadscLayout space and sets the map contents to the default.


   Input Parameters:
.    map - pointer to the map

   Level: developer

.seealso: PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetLocalSizes(), PetscThreadsLayout, 
          PetscThreadsLayoutDestroy(), PetscThreadsLayoutSetUp()
*/
PetscErrorCode PetscThreadsLayoutCreate(PetscThreadsLayout *tmap)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_PetscThreadsLayout,tmap);CHKERRQ(ierr);
  (*tmap)->nthreads = -1;
  (*tmap)->N        = -1;
  (*tmap)->trstarts =  0;
  (*tmap)->affinity =  0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutDestroy"
/*
     PetscThreadsLayoutDestroy - Frees a map object and frees its range if that exists.

   Input Parameters:
.    map - the PetscThreadsLayout

   Level: developer

      The PetscThreadsLayout object and methods are intended to be used in the PETSc threaded Vec and Mat implementions; it is 
      recommended they not be used in user codes unless you really gain something in their use.

.seealso: PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetLocalSizes(), PetscThreadsLayout, 
          PetscThreadsLayoutSetUp()
*/
PetscErrorCode PetscThreadsLayoutDestroy(PetscThreadsLayout *tmap)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if(!*tmap) PetscFunctionReturn(0);
  ierr = PetscFree((*tmap)->trstarts);CHKERRQ(ierr);
  ierr = PetscFree((*tmap)->affinity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetUp"
/*
     PetscThreadsLayoutSetUp - given a map where you have set the thread count, either global size or
           local sizes sets up the map so that it may be used.

   Input Parameters:
.    map - pointer to the map

   Level: developer

   Notes: Typical calling sequence
      PetscThreadsLayoutCreate(PetscThreadsLayout *);
      PetscThreadsLayoutSetNThreads(PetscThreadsLayout,nthreads);
      PetscThreadsLayoutSetSize(PetscThreadsLayout,N) or PetscThreadsLayoutSetLocalSizes(PetscThreadsLayout, *n); or both
      PetscThreadsLayoutSetUp(PetscThreadsLayout);

       If the local sizes, global size are already set and row offset exists then this does nothing.

.seealso: PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetLocalSizes(), PetscThreadsLayout, 
          PetscThreadsLayoutDestroy()
*/
PetscErrorCode PetscThreadsLayoutSetUp(PetscThreadsLayout tmap)
{
  PetscErrorCode     ierr;
  PetscInt           t,rstart=0,n,Q,R;
  PetscBool          S;
  
  PetscFunctionBegin;
  if(!tmap->nthreads) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Number of threads not set yet");
  if((tmap->N >= 0) && (tmap->trstarts)) PetscFunctionReturn(0);
  ierr = PetscMalloc((tmap->nthreads+1)*sizeof(PetscInt),&tmap->trstarts);CHKERRQ(ierr);

  Q = tmap->N/tmap->nthreads;
  R = tmap->N - Q*tmap->nthreads;
  for(t=0;t < tmap->nthreads;t++) {
    tmap->trstarts[t] = rstart;
    S               = (PetscBool)(t<R);
    n               = S?Q+1:Q;
    rstart         += n;
  }
  tmap->trstarts[tmap->nthreads] = rstart;
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutDuplicate"
/*@

    PetscThreadsLayoutDuplicate - creates a new PetscThreadsLayout with the same information as a given one. If the PetscThreadsLayout already exists it is destroyed first.

     Collective on PetscThreadsLayout

    Input Parameter:
.     in - input PetscThreadsLayout to be copied

    Output Parameter:
.     out - the copy

   Level: developer

    Notes: PetscThreadsLayoutSetUp() does not need to be called on the resulting PetscThreadsLayout

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutDestroy(), PetscThreadsLayoutSetUp()
@*/
PetscErrorCode PetscThreadsLayoutDuplicate(PetscThreadsLayout in,PetscThreadsLayout *out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadsLayoutDestroy(out);CHKERRQ(ierr);
  ierr = PetscThreadsLayoutCreate(out);CHKERRQ(ierr);
  ierr = PetscMemcpy(*out,in,sizeof(struct _n_PetscThreadsLayout));CHKERRQ(ierr);

  ierr = PetscMalloc(in->nthreads*sizeof(PetscInt),&(*out)->trstarts);CHKERRQ(ierr);
  ierr = PetscMemcpy((*out)->trstarts,in->trstarts,in->nthreads*sizeof(PetscInt));CHKERRQ(ierr);
  
  ierr = PetscMalloc(in->nthreads*sizeof(PetscInt),&(*out)->affinity);CHKERRQ(ierr);
  ierr = PetscMemcpy((*out)->affinity,in->affinity,in->nthreads*sizeof(PetscInt));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetLocalSizes"
/*
     PetscThreadsLayoutSetLocalSizes - Sets the local size for each thread 

   Input Parameters:
+    map - pointer to the map
-    n - local sizes

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutDestroy(), PetscThreadsLayoutGetLocalSizes()

*/
PetscErrorCode PetscThreadsLayoutSetLocalSizes(PetscThreadsLayout tmap,PetscInt n[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if(!tmap->nthreads) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Number of threads not set yet");
  ierr = PetscMalloc((tmap->nthreads+1)*sizeof(PetscInt),&tmap->trstarts);CHKERRQ(ierr);
  tmap->trstarts[0] = 0;
  for(i=1;i < tmap->nthreads+1;i++) tmap->trstarts[i] += n[i-1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutGetLocalSizes"
/*
     PetscThreadsLayoutGetLocalSizes - Gets the local size for each thread 

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - array to hold the local sizes (must be allocated)

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutDestroy(), PetscThreadsLayoutSetLocalSizes()
*/
PetscErrorCode PetscThreadsLayoutGetLocalSizes(PetscThreadsLayout tmap,PetscInt *n[])
{
  PetscInt i;
  PetscInt *tn=*n;
  PetscFunctionBegin;
  for(i=0;i < tmap->nthreads;i++) tn[i] = tmap->trstarts[i+1] - tmap->trstarts[i];
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetSize"
/*
     PetscThreadsLayoutSetSize - Sets the global size for PetscThreadsLayout object

   Input Parameters:
+    map - pointer to the map
-    n -   global size

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetSize()
*/
PetscErrorCode PetscThreadsLayoutSetSize(PetscThreadsLayout tmap,PetscInt N)
{
  PetscFunctionBegin;
  tmap->N = N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutGetSize"
/*
     PetscThreadsLayoutGetSize - Gets the global size 

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - global size

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutSetSize(), PetscThreadsLayoutGetLocalSizes()
*/
PetscErrorCode PetscThreadsLayoutGetSize(PetscThreadsLayout tmap,PetscInt *N)
{
  PetscFunctionBegin;
  *N = tmap->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetNThreads"
/*
     PetscThreadsLayoutSetNThreads - Sets the thread count for PetscThreadsLayout object

   Input Parameters:
+    map - pointer to the map
-    nthreads -   number of threads to be used with the map

   Level: developer

.seealso: PetscThreadsLayoutCreate(), PetscThreadsLayoutSetLocalSizes(), PetscThreadsLayoutGetSize()
*/
PetscErrorCode PetscThreadsLayoutSetNThreads(PetscThreadsLayout tmap,PetscInt nthreads)
{
  PetscFunctionBegin;
  tmap->nthreads = nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutSetThreadAffinities"
/*
     PetscThreadsLayoutSetLocalSizes - Sets the core affinities for PetscThreadsLayout object

   Input Parameters:
+    map - pointer to the map
-    affinities - core affinities for PetscThreadsLayout 

   Level: developer

.seealso: PetscThreadsLayoutGetThreadAffinities()

*/
PetscErrorCode PetscThreadsLayoutSetThreadAffinities(PetscThreadsLayout tmap, PetscInt affinities[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(tmap->nthreads*sizeof(PetscInt),&tmap->affinity);CHKERRQ(ierr);
  ierr = PetscMemcpy(tmap->affinity,affinities,tmap->nthreads*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadsLayoutGetThreadAffinities"
/*
     PetscThreadsLayoutGetThreadAffinities - Gets the core affinities of threads

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    affinity - core affinities of threads

   Level: developer

.seealso: PetscThreadsLayoutSetThreadAffinities()
*/
PetscErrorCode PetscThreadsLayoutGetThreadAffinities(PetscThreadsLayout tmap,const PetscInt *affinity[])
{
  PetscFunctionBegin;
  *affinity = tmap->affinity;
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
