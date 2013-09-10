#include <petsc-private/threadcommimpl.h>      /*I "petscthreadcomm.h" I*/
#include <petscviewer.h>
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

static PetscInt         N_CORES                          = -1;
PetscBool               PetscThreadCommRegisterAllCalled = PETSC_FALSE;
PetscFunctionList       PetscThreadCommList              = NULL;
PetscMPIInt             Petsc_ThreadComm_keyval          = MPI_KEYVAL_INVALID;
PetscThreadCommJobQueue PetscJobQueue                    = NULL;
PetscThreadComm         PETSC_THREAD_COMM_WORLD          = NULL;

/* Logging support */
PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;

static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...));

#undef __FUNCT__
#define __FUNCT__ "PetscGetNCores"
/*@
  PetscGetNCores - Gets the number of available cores on the system

  Not Collective

  Level: developer

  Notes
  Defaults to 1 if the available core count cannot be found

@*/
PetscErrorCode PetscGetNCores(PetscInt *ncores)
{
  PetscFunctionBegin;
  if (N_CORES == -1) {
    N_CORES = 1; /* Default value if number of cores cannot be found out */

#if defined(PETSC_HAVE_SYS_SYSINFO_H) && (PETSC_HAVE_GET_NPROCS) /* Linux */
    N_CORES = get_nprocs();
#elif defined(PETSC_HAVE_SYS_SYSCTL_H) && (PETSC_HAVE_SYSCTLBYNAME) /* MacOS, BSD */
    {
      PetscErrorCode ierr;
      size_t         len = sizeof(N_CORES);
      ierr = sysctlbyname("hw.activecpu",&N_CORES,&len,NULL,0); /* osx preferes activecpu over ncpu */
      if (ierr) { /* freebsd check ncpu */
        sysctlbyname("hw.ncpu",&N_CORES,&len,NULL,0);
        /* continue even if there is an error */
      }
    }
#elif defined(PETSC_HAVE_WINDOWS_H)   /* Windows */
    {
      SYSTEM_INFO sysinfo;
      GetSystemInfo(&sysinfo);
      N_CORES = sysinfo.dwNumberOfProcessors;
    }
#endif
  }
  if (ncores) *ncores = N_CORES;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscThreadCommWorldInitialize();
#undef __FUNCT__
#define __FUNCT__ "PetscGetThreadCommWorld"
/*
  PetscGetThreadCommWorld - Gets the global thread communicator.
                            Creates it if it does not exist already.

  Not Collective

  Output Parameters:
  tcommp - pointer to the global thread communicator

  Level: Intermediate
*/
PetscErrorCode PetscGetThreadCommWorld(PetscThreadComm *tcommp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PETSC_THREAD_COMM_WORLD) {
    ierr = PetscThreadCommWorldInitialize();CHKERRQ(ierr);
  }
  *tcommp = PETSC_THREAD_COMM_WORLD;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCommGetThreadComm"
/*@C
  PetscCommGetThreadComm - Gets the thread communicator
                           associated with the MPI communicator

  Not Collective

  Input Parameters:
. comm - the MPI communicator

  Output Parameters:
. tcommp - pointer to the thread communicator

  Notes: If no thread communicator is on the MPI_Comm then the global thread communicator
         is returned.
  Level: Intermediate

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy()
@*/
PetscErrorCode PetscCommGetThreadComm(MPI_Comm comm,PetscThreadComm *tcommp)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,(PetscThreadComm*)&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscGetThreadCommWorld(tcommp);CHKERRQ(ierr);
  } else *tcommp      = (PetscThreadComm)ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate"
/*
   PetscThreadCommCreate - Allocates a thread communicator object

   Not Collective

   Output Parameters:
.  tcomm - pointer to the thread communicator object

   Level: developer

.seealso: PetscThreadCommDestroy()
*/
PetscErrorCode PetscThreadCommCreate(PetscThreadComm *tcomm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcommout;

  PetscFunctionBegin;
  PetscValidPointer(tcomm,2);

  *tcomm = NULL;

  ierr                   = PetscNew(struct _p_PetscThreadComm,&tcommout);CHKERRQ(ierr);
  tcommout->refct        = 0;
  tcommout->nworkThreads =  -1;
  tcommout->affinities   = NULL;
  ierr                   = PetscNew(struct _PetscThreadCommOps,&tcommout->ops);CHKERRQ(ierr);
  tcommout->leader       = 0;
  *tcomm                 = tcommout;

  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)

static PetscErrorCode PetscThreadCommStackCreate_kernel(PetscInt trank)
{
  if (trank && !PetscStackActive()) {
    PetscStack *petscstack_in;
    petscstack_in = (PetscStack*)malloc(sizeof(PetscStack));
    petscstack_in->currentsize = 0;
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,petscstack_in);
  }
  return 0;
}

/* Creates stack frames for threads other than the main thread */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackCreate"
PetscErrorCode  PetscThreadCommStackCreate(void)
{
  PetscErrorCode ierr;
  ierr = PetscThreadCommRunKernel0(PETSC_COMM_SELF,(PetscThreadKernel)PetscThreadCommStackCreate_kernel);CHKERRQ(ierr);
  ierr = PetscThreadCommBarrier(PETSC_COMM_SELF);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode PetscThreadCommStackDestroy_kernel(PetscInt trank)
{
  if (trank && PetscStackActive()) {
    PetscStack *petscstack_in;
    petscstack_in = (PetscStack*)PetscThreadLocalGetValue(petscstack);
    free(petscstack_in);
    PetscThreadLocalSetValue((PetscThreadKey*)&petscstack,(PetscStack*)0);
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackDestroy"
/* Destroy stack frames for threads other than main thread
 *
 * The keyval may have been destroyed by the time this function is called, thus we must call
 * PetscThreadCommRunKernel0_Private so that we never reference an MPI_Comm.
 */
PetscErrorCode  PetscThreadCommStackDestroy(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscThreadCommRunKernel0_Private(PETSC_THREAD_COMM_WORLD,(PetscThreadKernel)PetscThreadCommStackDestroy_kernel);CHKERRQ(ierr);
  PETSC_THREAD_COMM_WORLD = NULL;
  PetscFunctionReturn(0);
  return 0;
}
#else
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackCreate"
PetscErrorCode  PetscThreadCommStackCreate(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommStackDestroy"
PetscErrorCode  PetscThreadCommStackDestroy(void)
{
  PetscFunctionBegin;
  PETSC_THREAD_COMM_WORLD = NULL;
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy"
/*
  PetscThreadCommDestroy - Frees a thread communicator object

  Not Collective

  Input Parameters:
. tcomm - the PetscThreadComm object

  Level: developer

.seealso: PetscThreadCommCreate()
*/
PetscErrorCode PetscThreadCommDestroy(PetscThreadComm *tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tcomm) PetscFunctionReturn(0);
  if (!--(*tcomm)->refct) {
    ierr = PetscThreadCommStackDestroy();CHKERRQ(ierr);
    /* Destroy the implementation specific data struct */
    if ((*tcomm)->ops->destroy) (*(*tcomm)->ops->destroy)(*tcomm);

    ierr = PetscFree((*tcomm)->affinities);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm)->ops);CHKERRQ(ierr);
    ierr = PetscFree(PetscJobQueue->jobs[0].job_status);CHKERRQ(ierr);
    ierr = PetscFree(PetscJobQueue->jobs);CHKERRQ(ierr);
    ierr = PetscFree(PetscJobQueue);CHKERRQ(ierr);
    ierr = PetscThreadCommReductionDestroy((*tcomm)->red);CHKERRQ(ierr);
    ierr = PetscFree((*tcomm));CHKERRQ(ierr);
  }
  *tcomm = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommView"
/*@C
   PetscThreadCommView - view a thread communicator

   Collective on comm

   Input Parameters:
+  comm - MPI communicator
-  viewer - viewer to display, for example PETSC_VIEWER_STDOUT_WORLD

   Level: developer

.seealso: PetscThreadCommCreate()
@*/
PetscErrorCode PetscThreadCommView(MPI_Comm comm,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  PetscBool       iascii;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Thread Communicator\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Number of threads = %D\n",tcomm->nworkThreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Type = %s\n",tcomm->type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if (tcomm->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*tcomm->ops->view)(tcomm,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetNThreads"
/*
   PetscThreadCommSetNThreads - Set the thread count for the thread communicator

   Not collective

   Input Parameters:
+  tcomm - the thread communicator
-  nthreads - Number of threads

   Options Database keys:
   -threadcomm_nthreads <nthreads> Number of threads to use

   Level: developer

   Notes:
   Defaults to using 1 thread.

   Use nthreads = PETSC_DECIDE or -threadcomm_nthreads PETSC_DECIDE for PETSc to decide the number of threads.


.seealso: PetscThreadCommGetNThreads()
*/
PetscErrorCode PetscThreadCommSetNThreads(PetscThreadComm tcomm,PetscInt nthreads)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       nthr;

  PetscFunctionBegin;
  if (nthreads == PETSC_DECIDE) {
    tcomm->nworkThreads = 1;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread comm - setting number of threads",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-threadcomm_nthreads","number of threads to use in the thread communicator","PetscThreadCommSetNThreads",1,&nthr,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      if (nthr == PETSC_DECIDE) tcomm->nworkThreads = N_CORES;
      else tcomm->nworkThreads = nthr;
    }
  } else tcomm->nworkThreads = nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetNThreads"
/*@C
   PetscThreadCommGetNThreads - Gets the thread count from the thread communicator
                                associated with the MPI communicator

   Not collective

   Input Parameters:
.  comm - the MPI communicator

   Output Parameters:
.  nthreads - number of threads

   Level: developer

.seealso: PetscThreadCommSetNThreads()
@*/
PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm comm,PetscInt *nthreads)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr      = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  *nthreads = tcomm->nworkThreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinities"
/*
   PetscThreadCommSetAffinities - Sets the core affinity for threads
                                  (which threads run on which cores)

   Not collective

   Input Parameters:
+  tcomm - the thread communicator
-  affinities - array of core affinity for threads

   Options Database keys:
.  -threadcomm_affinities <list of thread affinities>

   Level: developer

   Notes:
   Use affinities = NULL for PETSc to decide the affinities.
   If PETSc decides affinities, then each thread has affinity to
   a unique core with the main thread on Core 0, thread0 on core 1,
   and so on. If the thread count is more the number of available
   cores then multiple threads share a core.

   The first value is the affinity for the main thread

   The affinity list can be passed as
   a comma seperated list:                                 0,1,2,3,4,5,6,7
   a range (start-end+1):                                  0-8
   a range with given increment (start-end+1:inc):         0-7:2
   a combination of values and ranges seperated by commas: 0,1-8,8-15:2

   There must be no intervening spaces between the values.

.seealso: PetscThreadCommGetAffinities(), PetscThreadCommSetNThreads()
*/
PetscErrorCode PetscThreadCommSetAffinities(PetscThreadComm tcomm,const PetscInt affinities[])
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       nmax=tcomm->nworkThreads;

  PetscFunctionBegin;
  /* Free if affinities set already */
  ierr = PetscFree(tcomm->affinities);CHKERRQ(ierr);
  ierr = PetscMalloc(tcomm->nworkThreads*sizeof(PetscInt),&tcomm->affinities);CHKERRQ(ierr);

  if (!affinities) {
    /* Check if option is present in the options database */
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread comm - setting thread affinities",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-threadcomm_affinities","Set core affinities of threads","PetscThreadCommSetAffinities",tcomm->affinities,&nmax,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      if (nmax != tcomm->nworkThreads) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, Threads = %D, Core affinities set = %D",tcomm->nworkThreads,nmax);
    } else {
      /* PETSc default affinities */
      PetscInt i;
      for (i=0; i<tcomm->nworkThreads; i++) tcomm->affinities[i] = i%N_CORES;
    }
  } else {
    ierr = PetscMemcpy(tcomm->affinities,affinities,tcomm->nworkThreads*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetAffinities"
/*@C
   PetscThreadCommGetAffinities - Returns the core affinities set for the
                                  thread communicator associated with the MPI_Comm

    Not collective

    Input Parameters:
.   comm - MPI communicator

    Output Parameters:
.   affinities - thread affinities

    Level: developer

    Notes:
    The user must allocate space (nthreads PetscInts) for the
    affinities. Must call PetscThreadCommSetAffinities before.

*/
PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm comm,PetscInt affinities[])
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  PetscValidIntPointer(affinities,2);
  ierr = PetscMemcpy(affinities,tcomm->affinities,tcomm->nworkThreads*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetType"
/*
   PetscThreadCommSetType - Sets the threading model for the thread communicator

   Logically collective

   Input Parameters:
+  tcomm - the thread communicator
-  type  - the type of thread model needed


   Options Database keys:
   -threadcomm_type <type>

   Available types
   See "petsc/include/petscthreadcomm.h" for available types

*/
PetscErrorCode PetscThreadCommSetType(PetscThreadComm tcomm,PetscThreadCommType type)
{
  PetscErrorCode ierr,(*r)(PetscThreadComm);
  char           ttype[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidCharPointer(type,2);
  if (!PetscThreadCommRegisterAllCalled) { ierr = PetscThreadCommRegisterAll();CHKERRQ(ierr);}

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread comm - setting threading model",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadcomm_type","Thread communicator model","PetscThreadCommSetType",PetscThreadCommList,type,ttype,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscStrcpy(ttype,type);CHKERRQ(ierr);
  }
  ierr = PetscFunctionListFind(PetscThreadCommList,ttype,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PetscThreadComm type %s",ttype);
  ierr = (*r)(tcomm);CHKERRQ(ierr);
  ierr = PetscStrcmp(NOTHREAD,tcomm->type,&tcomm->isnothread);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier"
/*  PetscThreadCommBarrier - Apply a barrier on the thread communicator
                             associated with the MPI communicator

    Not collective

    Input Parameters:
.   comm - the MPI communicator

    Level: developer

    Notes:
    This routine provides an interface to put an explicit barrier between
    successive kernel calls to ensure that the first kernel is executed
    by all the threads before calling the next one.

    Called by the main thread only.

    May not be applicable to all types.
*/
PetscErrorCode PetscThreadCommBarrier(MPI_Comm comm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm->ops->barrier) {
    ierr = (*tcomm->ops->barrier)(tcomm);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegister"
/*@C
  PetscThreadCommRegister -

  Level: advanced
@*/
PetscErrorCode  PetscThreadCommRegister(const char sname[],PetscErrorCode (*function)(PetscThreadComm))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscThreadCommList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetScalars"
/*@C
   PetscThreadCommGetScalars - Gets pointers to locations for storing three PetscScalars that may be passed
                               to PetscThreadCommRunKernel to ensure that the scalar values remain valid
                               even after the main thread exits the calling function.

   Input Parameters:
+  comm - the MPI communicator having the thread communicator
.  val1 - pointer to store the first scalar value
.  val2 - pointer to store the second scalar value
-  val3 - pointer to store the third scalar value

   Level: developer

   Notes:
   This is a utility function to ensure that any scalars passed to PetscThreadCommRunKernel remain
   valid even after the main thread exits the calling function. If any scalars need to passed to
   PetscThreadCommRunKernel then these should be first stored in the locations provided by PetscThreadCommGetScalars()

   Pass NULL if any pointers are not needed.

   Called by the main thread only, not from within kernels

   Typical usage:

   PetscScalar *valptr;
   PetscThreadCommGetScalars(comm,&valptr,NULL,NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetScalars(MPI_Comm comm,PetscScalar **val1, PetscScalar **val2, PetscScalar **val3)
{
  PetscErrorCode        ierr;
  PetscThreadComm       tcomm;
  PetscThreadCommJobCtx job;
  PetscInt              job_num;

  PetscFunctionBegin;
  ierr    = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  job_num = PetscJobQueue->ctr%tcomm->nkernels;
  job     = &PetscJobQueue->jobs[job_num];
  if (val1) *val1 = &job->scalars[0];
  if (val2) *val2 = &job->scalars[1];
  if (val3) *val3 = &job->scalars[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetInts"
/*@C
   PetscThreadCommGetInts - Gets pointers to locations for storing three PetscInts that may be passed
                               to PetscThreadCommRunKernel to ensure that the scalar values remain valid
                               even after the main thread exits the calling function.

   Input Parameters:
+  comm - the MPI communicator having the thread communicator
.  val1 - pointer to store the first integer value
.  val2 - pointer to store the second integer value
-  val3 - pointer to store the third integer value

   Level: developer

   Notes:
   This is a utility function to ensure that any scalars passed to PetscThreadCommRunKernel remain
   valid even after the main thread exits the calling function. If any scalars need to passed to
   PetscThreadCommRunKernel then these should be first stored in the locations provided by PetscThreadCommGetInts()

   Pass NULL if any pointers are not needed.

   Called by the main thread only, not from within kernels

   Typical usage:

   PetscScalar *valptr;
   PetscThreadCommGetScalars(comm,&valptr,NULL,NULL);
   *valptr = alpha;   (alpha is the scalar you wish to pass in PetscThreadCommRunKernel)

   PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,valptr);

.seealso: PetscThreadCommRunKernel()
@*/
PetscErrorCode PetscThreadCommGetInts(MPI_Comm comm,PetscInt **val1, PetscInt **val2, PetscInt **val3)
{
  PetscErrorCode        ierr;
  PetscThreadComm       tcomm;
  PetscThreadCommJobCtx job;
  PetscInt              job_num;

  PetscFunctionBegin;
  ierr    = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  job_num = PetscJobQueue->ctr%tcomm->nkernels;
  job     = &PetscJobQueue->jobs[job_num];
  if (val1) *val1 = &job->ints[0];
  if (val2) *val2 = &job->ints[1];
  if (val3) *val3 = &job->ints[2];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel"
/*@C
   PetscThreadCommRunKernel - Runs the kernel using the thread communicator
                              associated with the MPI communicator

   Not Collective

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  nargs - Number of input arguments for the kernel
-  ...   - variable list of input arguments

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel(comm,(PetscThreadKernel)kernel_func,3,x,y,z);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x, PetscScalar* y, PetscReal* z)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),PetscInt nargs,...)
{
  PetscErrorCode        ierr;
  va_list               argptr;
  PetscInt              i;
  PetscThreadComm       tcomm=0;
  PetscThreadCommJobCtx job;

  PetscFunctionBegin;
  if (nargs > PETSC_KERNEL_NARGS_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Requested %D input arguments for kernel, max. limit %D",nargs,PETSC_KERNEL_NARGS_MAX);
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  job  = &PetscJobQueue->jobs[PetscJobQueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = PetscJobQueue->ctr;
  job->nargs          = nargs;
  job->pfunc          = (PetscThreadKernel)func;
  va_start(argptr,nargs);
  for (i=0; i < nargs; i++) job->args[i] = va_arg(argptr,void*);
  va_end(argptr);
  for (i=0; i<tcomm->nworkThreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  PetscJobQueue->ctr = (PetscJobQueue->ctr+1)%tcomm->nkernels; /* Increment the queue ctr to point to the next available slot */
  PetscJobQueue->kernel_ctr++;
  if (tcomm->isnothread) {
    ierr = PetscRunKernel(0,job->nargs,job);CHKERRQ(ierr);
    job->job_status[0] = THREAD_JOB_COMPLETED;
  } else {
    ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel0_Private"
/* The zero-argument kernel needs to be callable with an unwrapped PetscThreadComm after Petsc_ThreadComm_keyval has been freed. */
static PetscErrorCode PetscThreadCommRunKernel0_Private(PetscThreadComm tcomm,PetscErrorCode (*func)(PetscInt,...))
{
  PetscErrorCode        ierr;
  PetscInt              i;
  PetscThreadCommJobCtx job;

  PetscFunctionBegin;
  if (tcomm->isnothread) {
    ierr = (*func)(0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!PetscJobQueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &PetscJobQueue->jobs[PetscJobQueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = PetscJobQueue->ctr;
  job->nargs          = 1;
  job->pfunc          = (PetscThreadKernel)func;

  for (i=0; i<tcomm->nworkThreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  PetscJobQueue->ctr = (PetscJobQueue->ctr+1)%tcomm->nkernels; /* Increment the queue ctr to point to the next available slot */
  PetscJobQueue->kernel_ctr++;

  ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel0"
/*@C
   PetscThreadCommRunKernel0 - PetscThreadCommRunKernel version for kernels with no
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
-  func  - the kernel (needs to be cast to PetscThreadKernel)

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel0(comm,(PetscThreadKernel)kernel_func);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel0(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...))
{
  PetscErrorCode        ierr;
  PetscThreadComm       tcomm=0;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel0_Private(tcomm,func);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel1"
/*@C
   PetscThreadCommRunKernel1 - PetscThreadCommRunKernel version for kernels with 1
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
-  in1   - input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel1(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  PetscThreadComm       tcomm=0;
  PetscThreadCommJobCtx job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!PetscJobQueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &PetscJobQueue->jobs[PetscJobQueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = PetscJobQueue->ctr;
  job->nargs          = 1;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;

  for (i=0; i<tcomm->nworkThreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  PetscJobQueue->ctr = (PetscJobQueue->ctr+1)%tcomm->nkernels; /* Increment the queue ctr to point to the next available slot */
  PetscJobQueue->kernel_ctr++;

  ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel2"
/*@C
   PetscThreadCommRunKernel2 - PetscThreadCommRunKernel version for kernels with 2
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - 1st input argument for the kernel
-  in2   - 2nd input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt *x,PetscInt *y)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel2(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  PetscThreadComm       tcomm=0;
  PetscThreadCommJobCtx job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!PetscJobQueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &PetscJobQueue->jobs[PetscJobQueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = PetscJobQueue->ctr;
  job->nargs          = 2;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;

  for (i=0; i<tcomm->nworkThreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  PetscJobQueue->ctr = (PetscJobQueue->ctr+1)%tcomm->nkernels; /* Increment the queue ctr to point to the next available slot */
  PetscJobQueue->kernel_ctr++;

  ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel3"
/*@C
   PetscThreadCommRunKernel3 - PetscThreadCommRunKernel version for kernels with 3
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
-  in3   - third input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel3(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  PetscThreadComm       tcomm=0;
  PetscThreadCommJobCtx job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!PetscJobQueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &PetscJobQueue->jobs[PetscJobQueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = PetscJobQueue->ctr;
  job->nargs          = 3;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;
  job->args[2]        = in3;

  for (i=0; i<tcomm->nworkThreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  PetscJobQueue->ctr = (PetscJobQueue->ctr+1)%tcomm->nkernels; /* Increment the queue ctr to point to the next available slot */
  PetscJobQueue->kernel_ctr++;

  ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel4"
/*@C
   PetscThreadCommRunKernel4 - PetscThreadCommRunKernel version for kernels with 4
                               input argument

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
.  in3   - third input argument for the kernel
-  in4   - fourth input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel4(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3,void *in4)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  PetscThreadComm       tcomm=0;
  PetscThreadCommJobCtx job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3,in4);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!PetscJobQueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &PetscJobQueue->jobs[PetscJobQueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = PetscJobQueue->ctr;
  job->nargs          = 4;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;
  job->args[2]        = in3;
  job->args[3]        = in4;

  for (i=0; i<tcomm->nworkThreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  PetscJobQueue->ctr = (PetscJobQueue->ctr+1)%tcomm->nkernels; /* Increment the queue ctr to point to the next available slot */
  PetscJobQueue->kernel_ctr++;

  ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel6"
/*@C
   PetscThreadCommRunKernel6 - PetscThreadCommRunKernel version for kernels with 6
                               input arguments

   Input Parameters:
+  comm  - the MPI communicator
.  func  - the kernel (needs to be cast to PetscThreadKernel)
.  in1   - first input argument for the kernel
.  in2   - second input argument for the kernel
.  in3   - third input argument for the kernel
.  in4   - fourth input argument for the kernel
.  in5   - fifth input argument for the kernel
-  in6   - sixth input argument for the kernel

   Level: developer

   Notes:
   All input arguments to the kernel must be passed by reference, Petsc objects are
   inherrently passed by reference so you don't need to additionally & them.

   Example usage - PetscThreadCommRunKernel1(comm,(PetscThreadKernel)kernel_func,x);
   with kernel_func declared as
   PetscErrorCode kernel_func(PetscInt thread_id,PetscInt* x)

   The first input argument of kernel_func, thread_id, is the thread rank. This is passed implicitly
   by PETSc.

.seealso: PetscThreadCommCreate(), PetscThreadCommGNThreads()
@*/
PetscErrorCode PetscThreadCommRunKernel6(MPI_Comm comm,PetscErrorCode (*func)(PetscInt,...),void *in1,void *in2,void *in3,void *in4,void *in5,void *in6)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  PetscThreadComm       tcomm=0;
  PetscThreadCommJobCtx job;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm->isnothread) {
    ierr = (*func)(0,in1,in2,in3,in4,in5,in6);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (!PetscJobQueue) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to run kernel with no job queue");
  job = &PetscJobQueue->jobs[PetscJobQueue->ctr]; /* Get the job context from the queue to launch this job */
  if (job->job_status[0] != THREAD_JOB_NONE) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      while (PetscReadOnce(int,job->job_status[i]) != THREAD_JOB_COMPLETED) ;
    }
  }

  job->tcomm          = tcomm;
  job->tcomm->job_ctr = PetscJobQueue->ctr;
  job->nargs          = 6;
  job->pfunc          = (PetscThreadKernel)func;
  job->args[0]        = in1;
  job->args[1]        = in2;
  job->args[2]        = in3;
  job->args[3]        = in4;
  job->args[4]        = in5;
  job->args[5]        = in6;


  for (i=0; i<tcomm->nworkThreads; i++) job->job_status[i] = THREAD_JOB_POSTED;

  PetscJobQueue->ctr = (PetscJobQueue->ctr+1)%tcomm->nkernels; /* Increment the queue ctr to point to the next available slot */
  PetscJobQueue->kernel_ctr++;

  ierr = (*tcomm->ops->runkernel)(tcomm,job);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(ThreadComm_RunKernel,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Detaches the thread communicator from the MPI communicator if it exists
*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDetach"
PetscErrorCode PetscThreadCommDetach(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MPI_Attr_delete(comm,Petsc_ThreadComm_keyval);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   This routine attaches the thread communicator to the MPI communicator if it does not
   exist already.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAttach"
PetscErrorCode PetscThreadCommAttach(MPI_Comm comm,PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    tcomm->refct++;
    ierr = MPI_Attr_put(comm,Petsc_ThreadComm_keyval,tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommWorldInitialize"
/*
  PetscThreadCommWorldInitialize - Initializes the global thread communicator object

  PetscThreadCommWorldInitialize() defaults to using the nonthreaded communicator.
*/
PetscErrorCode PetscThreadCommWorldInitialize(void)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcomm;
  PetscInt        i,j;

  PetscFunctionBegin;
  ierr = PetscThreadCommCreate(&PETSC_THREAD_COMM_WORLD);CHKERRQ(ierr);
  tcomm = PETSC_THREAD_COMM_WORLD;
  ierr = PetscThreadCommSetNThreads(tcomm,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscThreadCommSetAffinities(tcomm,NULL);CHKERRQ(ierr);
  ierr = PetscNew(struct _p_PetscThreadCommJobQueue,&PetscJobQueue);CHKERRQ(ierr);

  tcomm->nkernels = 16;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread comm - setting number of kernels",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-threadcomm_nkernels","number of kernels that can be launched simultaneously","",16,&tcomm->nkernels,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscMalloc(tcomm->nkernels*sizeof(struct _p_PetscThreadCommJobCtx),&PetscJobQueue->jobs);CHKERRQ(ierr);
  ierr = PetscMalloc(tcomm->nworkThreads*tcomm->nkernels*sizeof(PetscInt),&PetscJobQueue->jobs[0].job_status);CHKERRQ(ierr);
  for (i=0; i<tcomm->nkernels; i++) {
    PetscJobQueue->jobs[i].job_status = PetscJobQueue->jobs[0].job_status + i*tcomm->nworkThreads;
    for (j=0; j<tcomm->nworkThreads; j++) PetscJobQueue->jobs[i].job_status[j] = THREAD_JOB_NONE;
  }
  PetscJobQueue->ctr        = 0;
  PetscJobQueue->kernel_ctr = 0;
  tcomm->job_ctr            = 0;

  ierr = PetscThreadCommSetType(tcomm,NOTHREAD);CHKERRQ(ierr);
  ierr = PetscThreadCommReductionCreate(tcomm,&tcomm->red);CHKERRQ(ierr);
  ierr = PetscThreadCommStackCreate();CHKERRQ(ierr);
  tcomm->refct++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetOwnershipRanges"
/*
   PetscThreadCommGetOwnershipRanges - Given the global size of an array, computes the local sizes and sets
                                       the starting array indices

   Input Parameters:
+  comm - the MPI communicator which holds the thread communicator
-  N    - the global size of the array

   Output Parameters:
.  trstarts - The starting array indices for each thread. the size of trstarts is nthreads+1

   Notes:
   trstarts is malloced in this routine
*/
PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm comm,PetscInt N,PetscInt *trstarts[])
{
  PetscErrorCode  ierr;
  PetscInt        Q,R;
  PetscBool       S;
  PetscThreadComm tcomm = NULL;
  PetscInt        *trstarts_out,nloc,i;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);

  ierr            = PetscMalloc((tcomm->nworkThreads+1)*sizeof(PetscInt),&trstarts_out);CHKERRQ(ierr);
  trstarts_out[0] = 0;
  Q               = N/tcomm->nworkThreads;
  R               = N - Q*tcomm->nworkThreads;
  for (i=0; i<tcomm->nworkThreads; i++) {
    S                 = (PetscBool)(i < R);
    nloc              = S ? Q+1 : Q;
    trstarts_out[i+1] = trstarts_out[i] + nloc;
  }

  *trstarts = trstarts_out;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetRank"
/*
   PetscThreadCommGetRank - Gets the rank of the calling thread

   Input Parameters:
.  tcomm - the thread communicator

   Output Parameters:
.  trank - The rank of the calling thread

*/
PetscErrorCode PetscThreadCommGetRank(PetscThreadComm tcomm,PetscInt *trank)
{
  PetscErrorCode ierr;
  PetscInt       rank = 0;

  PetscFunctionBegin;
  if (tcomm->ops->getrank) {
    ierr = (*tcomm->ops->getrank)(&rank);CHKERRQ(ierr);
  }
  *trank = rank;
  PetscFunctionReturn(0);
}
