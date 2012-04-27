#include <petsc-private/threadcommimpl.h>      /*I "petscthreadcomm.h" I*/

PetscInt N_CORES;

PetscBool  PetscThreadCommRegisterAllCalled = PETSC_FALSE;
PetscFList PetscThreadCommList              = PETSC_NULL;

static PetscBool  PetscGetNCoresCalled      = PETSC_FALSE;

PetscMPIInt Petsc_ThreadComm_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__
#define __FUNCT__ "PetscGetNCores"
/*
  PetscGetNCores - Gets the number of availalbe cores
                   on the system
		   
  Level: developer

  Notes
  Defaults to 1 if the available cores cannot be found
*/
PetscErrorCode PetscGetNCores(void)
{
  PetscFunctionBegin;
  N_CORES=1; /* Default value if N_CORES cannot be found out */
  /* Find the number of cores */
#if defined(PETSC_HAVE_SCHED_CPU_SET_T) /* Linux */
  N_CORES = get_nprocs();
#elif defined(PETSC_HAVE_SYS_SYSCTL_H) /* MacOS, BSD */
  {
    PetscErrorCode ierr;
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
  PetscFunctionReturn(0);
}
			    
#undef __FUNCT__
#define __FUNCT__ "PetscCommGetThreadComm"
/*
  PetscCommGetThreadComm - Gets the thread communicator
                           associated with the MPI communicator
  
  Input Parameters:
. comm - the MPI communicator

  Output Parameters:
. tcommp - pointer to the thread communicator
*/
PetscErrorCode PetscCommGetThreadComm(MPI_Comm comm,PetscThreadComm *tcommp)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void*          ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"MPI_Comm does not have a thread communicator");
  *tcommp = (PetscThreadComm)ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate"
/*
   PetscThreadCommCreate - Allocates a thread communicator object
 
   Input Parameters:
.  comm - the MPI communicator

   Output Parameters:
.  tcomm - pointer to the thread communicator object

   Level: developer

.seealso: PetscThreadCommDestroy()
*/
PetscErrorCode PetscThreadCommCreate(MPI_Comm comm,PetscThreadComm *tcomm)
{
  PetscErrorCode  ierr;
  PetscThreadComm tcommout;
  PetscInt        i;

  PetscFunctionBegin;
  PetscValidPointer(tcomm,2);
  *tcomm = PETSC_NULL;

  ierr = PetscNew(struct _p_PetscThreadComm,&tcommout);CHKERRQ(ierr);
  tcommout->nworkThreads =  -1;
  tcommout->affinities = PETSC_NULL;
  ierr = PetscNew(struct _PetscThreadCommOps,&tcommout->ops);CHKERRQ(ierr);
  ierr = PetscNew(struct _p_PetscThreadCommJobQueue,&tcommout->jobqueue);CHKERRQ(ierr);
  for(i=0;i<PETSC_KERNELS_MAX;i++) {
    ierr = PetscNew(struct _p_PetscThreadCommJobCtx,&tcommout->jobqueue->jobs[i]);CHKERRQ(ierr);
  }
  tcommout->jobqueue->ctr = 0;
  tcommout->leader = 0;
  *tcomm = tcommout;

  if(!PetscGetNCoresCalled) {     
    /* Set the number of available cores */
    ierr = PetscGetNCores();CHKERRQ(ierr);
    PetscGetNCoresCalled = PETSC_TRUE;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy"
/*
  PetscThreadCommDestroy - Frees a thread communicator object

  Input Parameters:
. tcomm - the PetscThreadComm object

  Level: developer

.seealso: PetscThreadCommCreate()
*/
PetscErrorCode PetscThreadCommDestroy(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if(!tcomm) PetscFunctionReturn(0);

  /* Destroy the implementation specific data struct */
  if(tcomm->ops->destroy) {
    (*tcomm->ops->destroy)(tcomm);
  } 

  ierr = PetscFree(tcomm->affinities);CHKERRQ(ierr);
  ierr = PetscFree(tcomm->ops);CHKERRQ(ierr);
  for(i=0;i<PETSC_KERNELS_MAX;i++) {
    ierr = PetscFree(tcomm->jobqueue->jobs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(tcomm->jobqueue);CHKERRQ(ierr);
  ierr = PetscFree(tcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommView"
/*@C
   PetscThreadCommView - view a thread communicator

   Collective

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
  if(!viewer) {ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);}
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Thread Communicator\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Number of threads = %D\n",tcomm->nworkThreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if(tcomm->ops->view) {
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
  PetscInt        nthr;
  PetscFunctionBegin;
  if(nthreads == PETSC_DECIDE) {
    tcomm->nworkThreads = 1;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Thread comm - setting number of threads",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-threadcomm_nthreads","number of threads to use in the thread communicator","PetscThreadCommSetNThreads",1,&nthr,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if(flg){ 
      if(nthr == PETSC_DECIDE) {
      tcomm->nworkThreads = N_CORES;
      } else tcomm->nworkThreads = nthr;
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
  PetscErrorCode ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
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
.  affinities - array of core affinity for threads

   Options Database keys:
   -thread_affinities <list of thread affinities> 

   Level: developer

   Notes:
   Use affinities = PETSC_NULL for PETSc to decide the affinities.
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

  if(affinities == PETSC_NULL) {
    /* Check if option is present in the options database */
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Thread comm - setting thread affinities",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-threadcomm_affinities","Set core affinities of threads","PetscThreadCommSetAffinities",tcomm->affinities,&nmax,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if(flg) {
      if(nmax != tcomm->nworkThreads) {
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, Threads = %D, Core affinities set = %D",tcomm->nworkThreads,nmax);
      }
    } else {
      /* PETSc default affinities */
      PetscInt i;
      for(i=0;i<tcomm->nworkThreads;i++) tcomm->affinities[i] = i%N_CORES;
    }
  } else {
    ierr = PetscMemcpy(tcomm->affinities,affinities,tcomm->nworkThreads*sizeof(PetscInt));
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
  PetscErrorCode ierr;
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
PetscErrorCode PetscThreadCommSetType(PetscThreadComm tcomm,const PetscThreadCommType type)
{
  PetscErrorCode ierr,(*r)(PetscThreadComm);
  char           ttype[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidCharPointer(type,2);
  if(!PetscThreadCommRegisterAllCalled) { ierr = PetscThreadCommRegisterAll(PETSC_NULL);CHKERRQ(ierr);}

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Thread comm - setting threading model",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsList("-threadcomm_type","Thread communicator model","PetscThreadCommSetType",PetscThreadCommList,PTHREAD,ttype,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscStrcpy(ttype,type);CHKERRQ(ierr); 
  }
  
  ierr = PetscFListFind(PetscThreadCommList,PETSC_COMM_WORLD,type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PetscThreadComm type %s",ttype);
  ierr = PetscStrcpy(tcomm->type,ttype);CHKERRQ(ierr);
  ierr = (*r)(tcomm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier"
/*  PetscThreadCommBarrier - Apply a barrier on the thread communicator
                             associated with the MPI communicator

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
  PetscErrorCode ierr;
  PetscThreadComm tcomm=0;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if(tcomm->ops->barrier) {
    (*tcomm->ops->barrier)(tcomm);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscThreadCommRegisterDestroy"
/*@C
   PetscThreadCommRegisterDestroy - Frees the list of thread communicator models that were
   registered by PetscThreadCommRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: PetscThreadComm, register, destroy

.seealso: PetscThreadCommRegisterAll()
@*/
PetscErrorCode  PetscThreadCommRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&PetscThreadCommList);CHKERRQ(ierr);
  PetscThreadCommRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscThreadCommRegister"
/*@C
  PetscThreadCommRegister - See PetscThreadCommRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  PetscThreadCommRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(PetscThreadComm))
{
  char           fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PetscThreadCommList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel"
/*@C
   PetscThreadCommRunKernel - Runs the kernel using the thread communicator
                              associated with the MPI communicator

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
  PetscErrorCode          ierr;
  va_list                 argptr;
  PetscInt                i;
  PetscThreadComm         tcomm=0;
  PetscThreadCommJobQueue queue;
  PetscThreadCommJobCtx   job;

  PetscFunctionBegin;
  if(nargs > PETSC_KERNEL_NARGS_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Requested %D input arguments for kernel, max. limit %D",nargs,PETSC_KERNEL_NARGS_MAX);
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  queue = tcomm->jobqueue;
  if(queue->ctr == PETSC_KERNELS_MAX) {
    /* Put a barrier so that the last given job is finished and reset the
       job queue counter
    */
    ierr = PetscThreadCommBarrier(comm);CHKERRQ(ierr);
    queue->ctr = 0;
  }
  job = queue->jobs[queue->ctr];
  job->tcomm = tcomm;
  job->nargs = nargs;
  job->pfunc = func;
  va_start(argptr,nargs);
  for(i=0; i < nargs; i++) {
    job->args[i] = va_arg(argptr,void*);
  }
  va_end(argptr);
  queue->ctr++;
  ierr = (*tcomm->ops->runkernel)(comm,job);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "Petsc_DelThreadComm"
/*
  This frees the thread communicator attached to MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()
*/
PetscMPIInt MPIAPI Petsc_DelThreadComm(MPI_Comm comm,PetscMPIInt keyval,void* tcomm,void* extra_state)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscThreadCommDestroy((PetscThreadComm)tcomm);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Deleting thread communicator data in an MPI_Comm %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize"
/*
  PetscThreadCommInitialize - Initializes the thread communicator object 
                              and stashes it inside PETSC_COMM_WORLD
                              
  PetscThreadCommInitialize() defaults to 1 thread and PTHREAD type.
*/
PetscErrorCode PetscThreadCommInitialize(void)
{
  PetscErrorCode ierr;
  PetscThreadComm tcomm;

  PetscFunctionBegin;
  if(Petsc_ThreadComm_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelThreadComm,&Petsc_ThreadComm_keyval,(void*)0);CHKERRQ(ierr);
  }
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,&tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommSetNThreads(tcomm,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscThreadCommSetAffinities(tcomm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Attr_put(PETSC_COMM_WORLD,Petsc_ThreadComm_keyval,(void*)tcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommSetType(tcomm,PTHREAD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
