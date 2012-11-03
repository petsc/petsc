
/*
   This file defines the initialization of PETSc, including PetscInitialize()
*/
#define PETSC_DESIRE_COMPLEX
#include <petscsys.h>        /*I  "petscsys.h"   I*/

#if defined(PETSC_HAVE_CUDA)
#include <cublas.h>
#endif

#include <petscthreadcomm.h>

#if defined(PETSC_USE_LOG)
extern PetscErrorCode PetscLogBegin_Private(void);
#endif
extern PetscBool  PetscHMPIWorker;

/* -----------------------------------------------------------------------------------------*/

extern FILE *petsc_history;

extern PetscErrorCode PetscInitialize_DynamicLibraries(void);
extern PetscErrorCode PetscFinalize_DynamicLibraries(void);
extern PetscErrorCode PetscFListDestroyAll(void);
extern PetscErrorCode PetscOpFListDestroyAll(void);
extern PetscErrorCode PetscSequentialPhaseBegin_Private(MPI_Comm,int);
extern PetscErrorCode PetscSequentialPhaseEnd_Private(MPI_Comm,int);
extern PetscErrorCode PetscCloseHistoryFile(FILE **);

/* user may set this BEFORE calling PetscInitialize() */
MPI_Comm PETSC_COMM_WORLD = MPI_COMM_NULL;

PetscMPIInt Petsc_Counter_keyval   = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_InnerComm_keyval = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_OuterComm_keyval = MPI_KEYVAL_INVALID;

/*
     Declare and set all the string names of the PETSc enums
*/
const char *const PetscBools[]     = {"FALSE","TRUE","PetscBool","PETSC_",0};
const char *const PetscCopyModes[] = {"COPY_VALUES","OWN_POINTER","USE_POINTER","PetscCopyMode","PETSC_",0};
const char *const PetscDataTypes[] = {"INT","DOUBLE","COMPLEX","LONG","SHORT","FLOAT",
                                "CHAR","LOGICAL","ENUM","BOOL","LONGDOUBLE","OBJECT","PetscDataType","PETSC_",0};

PetscBool  PetscPreLoadingUsed = PETSC_FALSE;
PetscBool  PetscPreLoadingOn   = PETSC_FALSE;

/* pthread_key for PetscStack */
#if defined(PETSC_HAVE_PTHREADCLASSES) && !defined(PETSC_PTHREAD_LOCAL)
pthread_key_t petscstack;
#endif

/*
       Checks the options database for initializations related to the
    PETSc components
*/
#undef __FUNCT__
#define __FUNCT__ "PetscOptionsCheckInitial_Components"
PetscErrorCode  PetscOptionsCheckInitial_Components(void)
{
  PetscBool  flg1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {
#if defined (PETSC_USE_LOG)
    MPI_Comm   comm = PETSC_COMM_WORLD;
    ierr = (*PetscHelpPrintf)(comm,"------Additional PETSc component options--------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_summary_exclude: <vec,mat,pc.ksp,snes>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -info_exclude: <null,vec,mat,pc,ksp,snes,ts>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

extern PetscBool PetscBeganMPI;

#undef __FUNCT__
#define __FUNCT__ "PetscInitializeNoPointers"
/*
      PetscInitializeNoPointers - Calls PetscInitialize() from C/C++ without the pointers to argc and args

   Collective

   Level: advanced

    Notes: this is called only by the PETSc MATLAB and Julia interface. Even though it might start MPI it sets the flag to
     indicate that it did NOT start MPI so that the PetscFinalize() does not end MPI, thus allowing PetscInitialize() to
     be called multiple times from MATLAB and Julia without the problem of trying to initialize MPI more than once.

     Turns off PETSc signal handling because that can interact with MATLAB's signal handling causing random crashes.

.seealso: PetscInitialize(), PetscInitializeFortran(), PetscInitializeNoArguments()
*/
PetscErrorCode  PetscInitializeNoPointers(int argc,char **args,const char *filename,const char *help)
{
  PetscErrorCode ierr;
  int            myargc = argc;
  char           **myargs = args;

  PetscFunctionBegin;
  ierr = PetscInitialize(&myargc,&myargs,filename,help);
  ierr = PetscPopSignalHandler();CHKERRQ(ierr);
  PetscBeganMPI = PETSC_FALSE;
  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGetPETSC_COMM_SELF"
/*
      Used by MATLAB and Julia interface to get communicator
*/
PetscErrorCode  PetscGetPETSC_COMM_SELF(MPI_Comm *comm)
{
  PetscFunctionBegin;
  *comm = PETSC_COMM_SELF;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscInitializeNoArguments"
/*@C
      PetscInitializeNoArguments - Calls PetscInitialize() from C/C++ without
        the command line arguments.

   Collective

   Level: advanced

.seealso: PetscInitialize(), PetscInitializeFortran()
@*/
PetscErrorCode  PetscInitializeNoArguments(void)
{
  PetscErrorCode ierr;
  int            argc = 0;
  char           **args = 0;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);
  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "PetscInitialized"
/*@
      PetscInitialized - Determine whether PETSc is initialized.

7   Level: beginner

.seealso: PetscInitialize(), PetscInitializeNoArguments(), PetscInitializeFortran()
@*/
PetscErrorCode  PetscInitialized(PetscBool  *isInitialized)
{
  PetscFunctionBegin;
  PetscValidPointer(isInitialized, 1);
  *isInitialized = PetscInitializeCalled;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFinalized"
/*@
      PetscFinalized - Determine whether PetscFinalize() has been called yet

   Level: developer

.seealso: PetscInitialize(), PetscInitializeNoArguments(), PetscInitializeFortran()
@*/
PetscErrorCode  PetscFinalized(PetscBool  *isFinalized)
{
  PetscFunctionBegin;
  PetscValidPointer(isFinalized, 1);
  *isFinalized = PetscFinalizeCalled;
  PetscFunctionReturn(0);
}

extern PetscErrorCode        PetscOptionsCheckInitial_Private(void);
extern PetscBool  PetscBeganMPI;

/*
       This function is the MPI reduction operation used to compute the sum of the
   first half of the datatype and the max of the second half.
*/
MPI_Op PetscMaxSum_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscMaxSum_Local"
void  MPIAPI PetscMaxSum_Local(void *in,void *out,int *cnt,MPI_Datatype *datatype)
{
  PetscInt *xin = (PetscInt*)in,*xout = (PetscInt*)out,i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_2INT) {
    (*PetscErrorPrintf)("Can only handle MPIU_2INT data types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  for (i=0; i<count; i++) {
    xout[2*i]    = PetscMax(xout[2*i],xin[2*i]);
    xout[2*i+1] += xin[2*i+1];
  }
  PetscFunctionReturnVoid();
}
EXTERN_C_END

/*
    Returns the max of the first entry owned by this processor and the
sum of the second entry.

    The reason nprocs[2*i] contains lengths nprocs[2*i+1] contains flag of 1 if length is nonzero
is so that the PetscMaxSum_Op() can set TWO values, if we passed in only nprocs[i] with lengths
there would be no place to store the both needed results.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscMaxSum"
PetscErrorCode  PetscMaxSum(MPI_Comm comm,const PetscInt nprocs[],PetscInt *max,PetscInt *sum)
{
  PetscMPIInt    size,rank;
  PetscInt       *work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr   = PetscMalloc(2*size*sizeof(PetscInt),&work);CHKERRQ(ierr);
  ierr   = MPI_Allreduce((void*)nprocs,work,size,MPIU_2INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
  *max   = work[2*rank];
  *sum   = work[2*rank+1];
  ierr   = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/
MPI_Op  PetscADMax_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscADMax_Local"
void  MPIAPI PetscADMax_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscScalar *xin = (PetscScalar *)in,*xout = (PetscScalar*)out;
  PetscInt    i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_2SCALAR) {
    (*PetscErrorPrintf)("Can only handle MPIU_2SCALAR data (i.e. double or complex) types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  for (i=0; i<count; i++) {
    if (PetscRealPart(xout[2*i]) < PetscRealPart(xin[2*i])) {
      xout[2*i]   = xin[2*i];
      xout[2*i+1] = xin[2*i+1];
    }
  }
  PetscFunctionReturnVoid();
}
EXTERN_C_END

MPI_Op  PetscADMin_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscADMin_Local"
void  MPIAPI PetscADMin_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscScalar *xin = (PetscScalar *)in,*xout = (PetscScalar*)out;
  PetscInt    i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_2SCALAR) {
    (*PetscErrorPrintf)("Can only handle MPIU_2SCALAR data (i.e. double or complex) types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  for (i=0; i<count; i++) {
    if (PetscRealPart(xout[2*i]) > PetscRealPart(xin[2*i])) {
      xout[2*i]   = xin[2*i];
      xout[2*i+1] = xin[2*i+1];
    }
  }
  PetscFunctionReturnVoid();
}
EXTERN_C_END
/* ---------------------------------------------------------------------------------------*/

#if (defined(PETSC_USE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128)
MPI_Op MPIU_SUM = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscSum_Local"
void  PetscSum_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscInt    i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_SCALAR) {
    PetscScalar *xin = (PetscScalar *)in,*xout = (PetscScalar*)out;
    for (i=0; i<count; i++) {
      xout[i] += xin[i];
    }
  } else if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal *)in,*xout = (PetscReal*)out;
    for (i=0; i<count; i++) {
      xout[i] += xin[i];
    }
  } else {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_SCALAR data (i.e. double or complex) types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  PetscFunctionReturnVoid();
}
EXTERN_C_END
#endif

#if defined(PETSC_USE_REAL___FLOAT128)
MPI_Op MPIU_MAX = 0;
MPI_Op MPIU_MIN = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscMax_Local"
void  PetscMax_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscInt    i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_SCALAR) {
    PetscScalar *xin = (PetscScalar *)in,*xout = (PetscScalar*)out;
    for (i=0; i<count; i++) {
      xout[i] = PetscRealPart(xout[i])<PetscRealPart(xin[i])? xin[i]: xout[i];
    }
  } else if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal *)in,*xout = (PetscReal*)out;
    for (i=0; i<count; i++) {
      xout[i] = PetscMax(xout[i],xin[i]);
    }
  } else {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_SCALAR data (i.e. double or complex) types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  PetscFunctionReturnVoid();
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscMin_Local"
void  PetscMin_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscInt    i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype == MPIU_SCALAR) {
    PetscScalar *xin = (PetscScalar *)in,*xout = (PetscScalar*)out;
    for (i=0; i<count; i++) {
      xout[i] = PetscRealPart(xout[i])>PetscRealPart(xin[i])? xin[i]: xout[i];
    }
  } else if (*datatype == MPIU_REAL) {
    PetscReal *xin = (PetscReal *)in,*xout = (PetscReal*)out;
    for (i=0; i<count; i++) {
      xout[i] = PetscMin(xout[i],xin[i]);
    }
  } else {
    (*PetscErrorPrintf)("Can only handle MPIU_REAL or MPIU_SCALAR data (i.e. double or complex) types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  PetscFunctionReturnVoid();
}
EXTERN_C_END
#endif

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "Petsc_DelCounter"
/*
   Private routine to delete internal tag/name counter storage when a communicator is freed.

   This is called by MPI, not by users. This is called by MPI_Comm_free() when the communicator that has this  data as an attribute is freed.

   Note: this is declared extern "C" because it is passed to MPI_Keyval_create()

*/
PetscMPIInt  MPIAPI Petsc_DelCounter(MPI_Comm comm,PetscMPIInt keyval,void *count_val,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting counter data in an MPI_Comm %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  ierr = PetscFree(count_val);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "Petsc_DelComm"
/*
  This does not actually free anything, it simply marks when a reference count to an internal or external MPI_Comm reaches zero and the
  the external MPI_Comm drops its reference to the internal or external MPI_Comm

  This is called by MPI, not by users. This is called when MPI_Comm_free() is called on the communicator.

  Note: this is declared extern "C" because it is passed to MPI_Keyval_create()

*/
PetscMPIInt  MPIAPI Petsc_DelComm(MPI_Comm comm,PetscMPIInt keyval,void *attr_val,void *extra_state)
{
  PetscErrorCode   ierr;
  PetscMPIInt      flg;
  MPI_Comm         icomm;
  void             *ptr;

  PetscFunctionBegin;
  ierr  = MPI_Attr_get(comm,Petsc_InnerComm_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    /*  Use PetscMemcpy() because casting from pointer to integer of different size is not allowed with some compilers  */
    ierr = PetscMemcpy(&icomm,&ptr,sizeof(MPI_Comm));CHKERRQ(ierr);
    ierr = MPI_Attr_get(icomm,Petsc_OuterComm_keyval,&ptr,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected reference to outer comm");
    ierr = MPI_Attr_delete(icomm,Petsc_OuterComm_keyval);CHKERRQ(ierr);
    ierr = PetscInfo1(0,"User MPI_Comm m %ld is being freed, removing reference from inner PETSc comm to this outer comm\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  } else {
    ierr = PetscInfo1(0,"Removing reference to PETSc communicator imbedded in a user MPI_Comm m %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  }
  PetscFunctionReturn(MPI_SUCCESS);
}
EXTERN_C_END

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
#if !defined(PETSC_WORDS_BIGENDIAN)
EXTERN_C_BEGIN
extern PetscMPIInt PetscDataRep_extent_fn(MPI_Datatype,MPI_Aint*,void*);
extern PetscMPIInt PetscDataRep_read_conv_fn(void*, MPI_Datatype,PetscMPIInt,void*,MPI_Offset,void*);
extern PetscMPIInt PetscDataRep_write_conv_fn(void*, MPI_Datatype,PetscMPIInt,void*,MPI_Offset,void*);
EXTERN_C_END
#endif
#endif

int  PetscGlobalArgc   = 0;
char **PetscGlobalArgs = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscGetArgs"
/*@C
   PetscGetArgs - Allows you to access the raw command line arguments anywhere
     after PetscInitialize() is called but before PetscFinalize().

   Not Collective

   Output Parameters:
+  argc - count of number of command line arguments
-  args - the command line arguments

   Level: intermediate

   Notes:
      This is usually used to pass the command line arguments into other libraries
   that are called internally deep in PETSc or the application.

      The first argument contains the program name as is normal for C arguments.

   Concepts: command line arguments

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArguments()

@*/
PetscErrorCode  PetscGetArgs(int *argc,char ***args)
{
  PetscFunctionBegin;
  if (!PetscInitializeCalled && PetscFinalizeCalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"You must call after PetscInitialize() but before PetscFinalize()");
  *argc = PetscGlobalArgc;
  *args = PetscGlobalArgs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGetArguments"
/*@C
   PetscGetArguments - Allows you to access the  command line arguments anywhere
     after PetscInitialize() is called but before PetscFinalize().

   Not Collective

   Output Parameters:
.  args - the command line arguments

   Level: intermediate

   Notes:
      This does NOT start with the program name and IS null terminated (final arg is void)

   Concepts: command line arguments

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscFreeArguments()

@*/
PetscErrorCode  PetscGetArguments(char ***args)
{
  PetscInt       i,argc = PetscGlobalArgc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!PetscInitializeCalled && PetscFinalizeCalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"You must call after PetscInitialize() but before PetscFinalize()");
  if (!argc) {*args = 0; PetscFunctionReturn(0);}
  ierr = PetscMalloc(argc*sizeof(char*),args);CHKERRQ(ierr);
  for (i=0; i<argc-1; i++) {
    ierr = PetscStrallocpy(PetscGlobalArgs[i+1],&(*args)[i]);CHKERRQ(ierr);
  }
  (*args)[argc-1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFreeArguments"
/*@C
   PetscFreeArguments - Frees the memory obtained with PetscGetArguments()

   Not Collective

   Output Parameters:
.  args - the command line arguments

   Level: intermediate

   Concepts: command line arguments

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscGetArguments()

@*/
PetscErrorCode  PetscFreeArguments(char **args)
{
  PetscInt       i = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!args) {PetscFunctionReturn(0);}
  while (args[i]) {
    ierr = PetscFree(args[i]);CHKERRQ(ierr);
    i++;
  }
  ierr = PetscFree(args);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscInitialize"
/*@C
   PetscInitialize - Initializes the PETSc database and MPI.
   PetscInitialize() calls MPI_Init() if that has yet to be called,
   so this routine should always be called near the beginning of
   your program -- usually the very first line!

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, also checks ~username/.petscrc and .petscrc use PETSC_NULL to not check for
          code specific file. Use -skip_petscrc in the code specific file to skip the .petscrc files
-  help - [optional] Help message to print, use PETSC_NULL for no message

   If you wish PETSc code to run ONLY on a subcommunicator of MPI_COMM_WORLD, create that
   communicator first and assign it to PETSC_COMM_WORLD BEFORE calling PetscInitialize(). Thus if you are running a
   four process job and two processes will run PETSc and have PetscInitialize() and PetscFinalize() and two process will not,
   then do this. If ALL processes in the job are using PetscInitialize() and PetscFinalize() then you don't need to do this, even
   if different subcommunicators of the job are doing different things with PETSc.

   Options Database Keys:
+  -start_in_debugger [noxterm,dbx,xdb,gdb,...] - Starts program in debugger
.  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -on_error_emacs <machinename> causes emacsclient to jump to error file
.  -on_error_abort calls abort() when error detected (no traceback)
.  -on_error_mpiabort calls MPI_abort() when error detected
.  -error_output_stderr prints error messages to stderr instead of the default stdout
.  -error_output_none does not print the error messages (but handles errors in the same way as if this was not called)
.  -debugger_nodes [node1,node2,...] - Indicates nodes to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -stop_for_debugger - Print message on how to attach debugger manually to
                        process and wait (-debugger_pause) seconds for attachment
.  -malloc - Indicates use of PETSc error-checking malloc (on by default for debug version of libraries)
.  -malloc no - Indicates not to use error-checking malloc
.  -malloc_debug - check for memory corruption at EVERY malloc or free
.  -malloc_test - like -malloc_dump -malloc_debug, but only active for debugging builds
.  -fp_trap - Stops on floating point exceptions (Note that on the
              IBM RS6000 this slows code by at least a factor of 10.)
.  -no_signal_handler - Indicates not to trap error signals
.  -shared_tmp - indicates /tmp directory is shared by all processors
.  -not_shared_tmp - each processor has own /tmp
.  -tmp - alternative name of /tmp directory
.  -get_total_flops - returns total flops done by all processors
.  -memory_info - Print memory usage at end of run
-  -server <port> - start PETSc webserver (default port is 8080)

   Options Database Keys for Profiling:
   See the <a href="../../docs/manual.pdf#nameddest=ch_profiling">profiling chapter of the users manual</a> for details.
+  -log_trace [filename] - Print traces of all PETSc calls
        to the screen (useful to determine where a program
        hangs without running in the debugger).  See PetscLogTraceBegin().
.  -info <optional filename> - Prints verbose information to the screen
-  -info_exclude <null,vec,mat,pc,ksp,snes,ts> - Excludes some of the verbose messages

   Environmental Variables:
+   PETSC_TMP - alternative tmp directory
.   PETSC_SHARED_TMP - tmp is shared by all processes
.   PETSC_NOT_SHARED_TMP - each process has its own private tmp
.   PETSC_VIEWER_SOCKET_PORT - socket number to use for socket viewer
-   PETSC_VIEWER_SOCKET_MACHINE - machine to use for socket viewer to connect to


   Level: beginner

   Notes:
   If for some reason you must call MPI_Init() separately, call
   it before PetscInitialize().

   Fortran Version:
   In Fortran this routine has the format
$       call PetscInitialize(file,ierr)

+   ierr - error return code
-  file - [optional] PETSc database file, also checks ~username/.petscrc and .petscrc use PETSC_NULL_CHARACTER to not check for
          code specific file. Use -skip_petscrc in the code specific file to skip the .petscrc files

   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as
   in the C version. See the <a href="../../docs/manual.pdf">users manual</a> for details.

   If your main program is C but you call Fortran code that also uses PETSc you need to call PetscInitializeFortran() soon after
   calling PetscInitialize().

   Concepts: initializing PETSc

.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscInitializeNoArguments()

@*/
PetscErrorCode  PetscInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscErrorCode ierr;
  PetscMPIInt    flag, size;
  PetscInt       nodesize;
  PetscBool      flg;
  char           hostname[256];

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(0);

  /* these must be initialized in a routine, not as a constant declaration*/
  PETSC_STDOUT = stdout;
  PETSC_STDERR = stderr;

  ierr = PetscOptionsCreate();CHKERRQ(ierr);

  /*
     We initialize the program name here (before MPI_Init()) because MPICH has a bug in
     it that it sets args[0] on all processors to be args[0] on the first processor.
  */
  if (argc && *argc) {
    ierr = PetscSetProgramName(**args);CHKERRQ(ierr);
  } else {
    ierr = PetscSetProgramName("Unknown Name");CHKERRQ(ierr);
  }

  ierr = MPI_Initialized(&flag);CHKERRQ(ierr);
  if (!flag) {
    if (PETSC_COMM_WORLD != MPI_COMM_NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"You cannot set PETSC_COMM_WORLD if you have not initialized MPI first");
#if defined(PETSC_HAVE_MPI_INIT_THREAD)
    {
      PetscMPIInt provided;
      ierr = MPI_Init_thread(argc,args,MPI_THREAD_FUNNELED,&provided);CHKERRQ(ierr);
    }
#else
    ierr = MPI_Init(argc,args);CHKERRQ(ierr);
#endif
    PetscBeganMPI = PETSC_TRUE;
  }
  if (argc && args) {
    PetscGlobalArgc = *argc;
    PetscGlobalArgs = *args;
  }
  PetscFinalizeCalled   = PETSC_FALSE;

  if (PETSC_COMM_WORLD == MPI_COMM_NULL) {
    PETSC_COMM_WORLD = MPI_COMM_WORLD;
  }
  ierr = MPI_Errhandler_set(PETSC_COMM_WORLD,MPI_ERRORS_RETURN);CHKERRQ(ierr);

  /* Done after init due to a bug in MPICH-GM? */
  ierr = PetscErrorPrintfInitialize();CHKERRQ(ierr);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);CHKERRQ(ierr);

  /*
     Initialized the global complex variable; this is because with
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
#if defined(PETSC_HAVE_COMPLEX)
  {
#if defined(PETSC_CLANGUAGE_CXX)
    PetscComplex ic(0.0,1.0);
    PETSC_i = ic;
#elif defined(PETSC_CLANGUAGE_C)
    PETSC_i = _Complex_I;
#endif
  }

#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
  ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU_C_DOUBLE_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_C_DOUBLE_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_contiguous(2,MPI_FLOAT,&MPIU_C_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_C_COMPLEX);CHKERRQ(ierr);
#endif
#endif /* PETSC_HAVE_COMPLEX */

  /*
     Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  ierr = MPI_Op_create(PetscMaxSum_Local,1,&PetscMaxSum_Op);CHKERRQ(ierr);

#if defined(PETSC_USE_REAL___FLOAT128)
  ierr = MPI_Type_contiguous(2,MPI_DOUBLE,&MPIU___FLOAT128);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU___FLOAT128);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = MPI_Type_contiguous(4,MPI_DOUBLE,&MPIU___COMPLEX128);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU___COMPLEX128);CHKERRQ(ierr);
#endif
  ierr = MPI_Op_create(PetscMax_Local,1,&MPIU_MAX);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscMin_Local,1,&MPIU_MIN);CHKERRQ(ierr);
#endif

#if (defined(PETSC_USE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128)
  ierr = MPI_Op_create(PetscSum_Local,1,&MPIU_SUM);CHKERRQ(ierr);
#endif

  ierr = MPI_Type_contiguous(2,MPIU_SCALAR,&MPIU_2SCALAR);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_2SCALAR);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscADMax_Local,1,&PetscADMax_Op);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscADMin_Local,1,&PetscADMin_Op);CHKERRQ(ierr);

  ierr = MPI_Type_contiguous(2,MPIU_INT,&MPIU_2INT);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_2INT);CHKERRQ(ierr);

  /*
     Attributes to be set on PETSc communicators
  */
  ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelCounter,&Petsc_Counter_keyval,(void*)0);CHKERRQ(ierr);
  ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelComm,&Petsc_InnerComm_keyval,(void*)0);CHKERRQ(ierr);
  ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelComm,&Petsc_OuterComm_keyval,(void*)0);CHKERRQ(ierr);

  /*
     Build the options database
  */
  ierr = PetscOptionsInsert(argc,args,file);CHKERRQ(ierr);


  /*
     Print main application help message
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (help && flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,help);CHKERRQ(ierr);
  }
  ierr = PetscOptionsCheckInitial_Private();CHKERRQ(ierr);

  /* SHOULD PUT IN GUARDS: Make sure logging is initialized, even if we do not print it out */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogBegin_Private();CHKERRQ(ierr);
#endif

  /*
     Load the dynamic libraries (on machines that support them), this registers all
     the solvers etc. (On non-dynamic machines this initializes the PetscDraw and PetscViewer classes)
  */
  ierr = PetscInitialize_DynamicLibraries();CHKERRQ(ierr);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"PETSc successfully started: number of processors = %d\n",size);CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,256);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Running on machine: %s\n",hostname);CHKERRQ(ierr);

  ierr = PetscOptionsCheckInitial_Components();CHKERRQ(ierr);
  /* Check the options database for options related to the options database itself */
  ierr = PetscOptionsSetFromOptions();CHKERRQ(ierr);

#if defined(PETSC_USE_PETSC_MPI_EXTERNAL32)
  /*
      Tell MPI about our own data representation converter, this would/should be used if extern32 is not supported by the MPI

      Currently not used because it is not supported by MPICH.
  */
#if !defined(PETSC_WORDS_BIGENDIAN)
  ierr = MPI_Register_datarep((char *)"petsc",PetscDataRep_read_conv_fn,PetscDataRep_write_conv_fn,PetscDataRep_extent_fn,PETSC_NULL);CHKERRQ(ierr);
#endif
#endif

  ierr = PetscOptionsGetInt(PETSC_NULL,"-hmpi_spawn_size",&nodesize,&flg);CHKERRQ(ierr);
  if (flg) {
#if defined(PETSC_HAVE_MPI_COMM_SPAWN)
    ierr = PetscHMPISpawn((PetscMPIInt) nodesize);CHKERRQ(ierr); /* worker nodes never return from here; they go directly to PetscEnd() */
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc built without MPI 2 (MPI_Comm_spawn) support, use -hmpi_merge_size instead");
#endif
  } else {
    ierr = PetscOptionsGetInt(PETSC_NULL,"-hmpi_merge_size",&nodesize,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscHMPIMerge((PetscMPIInt) nodesize,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      if (PetscHMPIWorker) { /* if worker then never enter user code */
        PetscInitializeCalled = PETSC_TRUE;
        ierr = PetscEnd();
      }
    }
  }

#if defined(PETSC_HAVE_CUDA)
  {
    PetscMPIInt p;
    for (p = 0; p < PetscGlobalSize; ++p) {
      if (p == PetscGlobalRank) {cublasInit();}
      ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    }
  }
#endif

#if defined(PETSC_HAVE_AMS)
  ierr = PetscOptionsHasName(PETSC_NULL,"-ams_publish_objects",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscAMSPublishAll = PETSC_TRUE;
  }
#endif

  ierr = PetscOptionsHasName(PETSC_NULL,"-python",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscInitializeCalled = PETSC_TRUE;
    ierr = PetscPythonInitialize(PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscThreadCommInitializePackage(PETSC_NULL);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  PetscThreadLocalRegister(&petscstack); /* Creates petscstack_key if needed */
  ierr = PetscStackCreate();CHKERRQ(ierr);
#endif

  /*
      Once we are completedly initialized then we can set this variables
  */
  PetscInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

extern PetscObject *PetscObjects;
extern PetscInt    PetscObjectsCounts, PetscObjectsMaxCounts;

#undef __FUNCT__
#define __FUNCT__ "PetscFinalize"
/*@C
   PetscFinalize - Checks for options to be called at the conclusion
   of the program. MPI_Finalize() is called only if the user had not
   called MPI_Init() before calling PetscInitialize().

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -options_table - Calls PetscOptionsView()
.  -options_left - Prints unused options that remain in the database
.  -objects_left  - Prints list of all objects that have not been freed
.  -mpidump - Calls PetscMPIDump()
.  -malloc_dump - Calls PetscMallocDump()
.  -malloc_info - Prints total memory usage
-  -malloc_log - Prints summary of memory usage

   Options Database Keys for Profiling:
   See the <a href="../../docs/manual.pdf#nameddest=ch_profiling">profiling chapter of the users manual</a> for details.
+  -log_summary [filename] - Prints summary of flop and timing
        information to screen. If the filename is specified the
        summary is written to the file.  See PetscLogView().
.  -log_summary_python [filename] - Prints data on of flop and timing usage to a file or screen.
        See PetscLogPrintSViewPython().
.  -log_all [filename] - Logs extensive profiling information
        See PetscLogDump().
.  -log [filename] - Logs basic profiline information  See PetscLogDump().
.  -log_sync - Log the synchronization in scatters, inner products
        and norms
-  -log_mpe [filename] - Creates a logfile viewable by the
      utility Upshot/Nupshot (in MPICH distribution)

   Level: beginner

   Note:
   See PetscInitialize() for more general runtime options.

.seealso: PetscInitialize(), PetscOptionsView(), PetscMallocDump(), PetscMPIDump(), PetscEnd()
@*/
PetscErrorCode  PetscFinalize(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,nopt;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE,objects_left = PETSC_FALSE;
#if defined(PETSC_HAVE_AMS)
  PetscBool      flg = PETSC_FALSE;
#endif
#if defined(PETSC_USE_LOG)
  char           mname[PETSC_MAX_PATH_LEN];
#endif

  PetscFunctionBegin;

  if (!PetscInitializeCalled) {
    printf("PetscInitialize() must be called before PetscFinalize()\n");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  ierr = PetscInfo(PETSC_NULL,"PetscFinalize() called\n");

#if defined(PETSC_HAVE_AMS)
  ierr = PetscOptionsGetBool(PETSC_NULL,"-options_gui",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscOptionsAMSDestroy();CHKERRQ(ierr);
  }
#endif

  ierr = PetscHMPIFinalize();CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc_info",&flg2,PETSC_NULL);CHKERRQ(ierr);
  if (!flg2) {
    flg2 = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL,"-memory_info",&flg2,PETSC_NULL);CHKERRQ(ierr);
  }
  if (flg2) {
    ierr = PetscMemoryShowUsage(PETSC_VIEWER_STDOUT_WORLD,"Summary of Memory Usage in PETSc\n");CHKERRQ(ierr);
  }

#if defined(PETSC_USE_LOG)
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-get_total_flops",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) {
    PetscLogDouble flops = 0;
    ierr = MPI_Reduce(&petsc_TotalFlops,&flops,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Total flops over all processors %g\n",flops);CHKERRQ(ierr);
  }
#endif


#if defined(PETSC_USE_LOG)
#if defined(PETSC_HAVE_MPE)
  mname[0] = 0;
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_mpe",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1){
    if (mname[0]) {ierr = PetscLogMPEDump(mname);CHKERRQ(ierr);}
    else          {ierr = PetscLogMPEDump(0);CHKERRQ(ierr);}
  }
#endif
  mname[0] = 0;
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_summary",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscViewer viewer;
    if (mname[0])  {
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,mname,&viewer);CHKERRQ(ierr);
      ierr = PetscLogView(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      viewer = PETSC_VIEWER_STDOUT_WORLD;
      ierr = PetscLogView(viewer);CHKERRQ(ierr);
    }
  }

  mname[0] = 0;
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_summary_python",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscViewer viewer;
    if (mname[0])  {
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,mname,&viewer);CHKERRQ(ierr);
      ierr = PetscLogViewPython(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      viewer = PETSC_VIEWER_STDOUT_WORLD;
      ierr = PetscLogViewPython(viewer);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-log_detailed",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1) {
    if (mname[0])  {ierr = PetscLogPrintDetailed(PETSC_COMM_WORLD,mname);CHKERRQ(ierr);}
    else           {ierr = PetscLogPrintDetailed(PETSC_COMM_WORLD,0);CHKERRQ(ierr);}
  }

  mname[0] = 0;
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_all",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-log",mname,PETSC_MAX_PATH_LEN,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2){
    if (mname[0]) PetscLogDump(mname);
    else          PetscLogDump(0);
  }
#endif

#if defined(PETSC_USE_DEBUG)
  ierr = PetscStackDestroy();CHKERRQ(ierr);
#endif

  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-no_signal_handler",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (!flg1) { ierr = PetscPopSignalHandler();CHKERRQ(ierr);}
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mpidump",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscMPIDump(stdout);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  flg2 = PETSC_FALSE;
  /* preemptive call to avoid listing this option in options table as unused */
  ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_dump",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-options_table",&flg2,PETSC_NULL);CHKERRQ(ierr);

  if (flg2) {
    ierr = PetscOptionsView(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* to prevent PETSc -options_left from warning */
  ierr = PetscOptionsHasName(PETSC_NULL,"-nox",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-nox_warning",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-objects_left",&objects_left,PETSC_NULL);CHKERRQ(ierr);

  if (!PetscHMPIWorker) { /* worker processes skip this because they do not usually process options */
    flg3 = PETSC_FALSE; /* default value is required */
    ierr = PetscOptionsGetBool(PETSC_NULL,"-options_left",&flg3,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsAllUsed(&nopt);CHKERRQ(ierr);
    if (flg3) {
      if (!flg2) { /* have not yet printed the options */
	ierr = PetscOptionsView(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      }
      if (!nopt) {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"There are no unused options.\n");CHKERRQ(ierr);
      } else if (nopt == 1) {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"There is one unused database option. It is:\n");CHKERRQ(ierr);
      } else {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"There are %D unused database options. They are:\n",nopt);CHKERRQ(ierr);
      }
    }
#if defined(PETSC_USE_DEBUG)
    if (nopt && !flg3 && !flg1) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! There are options you set that were not used!\n");CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! could be spelling mistake, etc!\n");CHKERRQ(ierr);
      ierr = PetscOptionsLeft();CHKERRQ(ierr);
    } else if (nopt && flg3) {
#else
    if (nopt && flg3) {
#endif
      ierr = PetscOptionsLeft();CHKERRQ(ierr);
    }
  }

  /*
     Free all objects registered with PetscObjectRegisterDestroy() such as PETSC_VIEWER_XXX_().
  */
  ierr = PetscObjectRegisterDestroyAll();CHKERRQ(ierr);

  /*
       List all objects the user may have forgot to free
  */
  if (objects_left && PetscObjectsCounts) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The following objects %D were never freed\n",PetscObjectsCounts);
  }
  for (i=0; i<PetscObjectsMaxCounts; i++) {
    if (PetscObjects[i]) {
      if (objects_left) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"  %s %s %s\n",PetscObjects[i]->class_name,PetscObjects[i]->type_name,PetscObjects[i]->name);CHKERRQ(ierr);
      }
    }
  }
  /* cannot actually destroy the left over objects, but destroy the list */
  PetscObjectsCounts    = 0;
  PetscObjectsMaxCounts = 0;
  ierr = PetscFree(PetscObjects);CHKERRQ(ierr);


#if defined(PETSC_USE_LOG)
  ierr = PetscLogDestroy();CHKERRQ(ierr);
#endif

  /*
       Free all the registered create functions, such as KSPList, VecList, SNESList, etc
  */
  ierr = PetscFListDestroyAll();CHKERRQ(ierr);

  /*
       Free all the registered op functions, such as MatOpList, etc
  */
  ierr = PetscOpFListDestroyAll();CHKERRQ(ierr);

  /*
     Destroy any packages that registered a finalize
  */
  ierr = PetscRegisterFinalizeAll();CHKERRQ(ierr);

  /*
     Destroy all the function registration lists created
  */
  ierr = PetscFinalize_DynamicLibraries();CHKERRQ(ierr);

  if (petsc_history) {
    ierr = PetscCloseHistoryFile(&petsc_history);CHKERRQ(ierr);
    petsc_history = 0;
  }

  ierr = PetscInfoAllow(PETSC_FALSE,PETSC_NULL);CHKERRQ(ierr);

  {
    char fname[PETSC_MAX_PATH_LEN];
    FILE *fd;
    int  err;

    fname[0] = 0;
    ierr = PetscOptionsGetString(PETSC_NULL,"-malloc_dump",fname,250,&flg1);CHKERRQ(ierr);
    flg2 = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc_test",&flg2,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    if (PETSC_RUNNING_ON_VALGRIND) flg2 = PETSC_FALSE;
#else
    flg2 = PETSC_FALSE;         /* Skip reporting for optimized builds regardless of -malloc_test */
#endif
    if (flg1 && fname[0]) {
      char sname[PETSC_MAX_PATH_LEN];

      sprintf(sname,"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",sname);
      ierr = PetscMallocDump(fd);CHKERRQ(ierr);
      err = fclose(fd);
      if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
    } else if (flg1 || flg2) {
      MPI_Comm local_comm;

      ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseBegin_Private(local_comm,1);CHKERRQ(ierr);
        ierr = PetscMallocDump(stdout);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseEnd_Private(local_comm,1);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
    }
  }
  {
    char fname[PETSC_MAX_PATH_LEN];
    FILE *fd = PETSC_NULL;

    fname[0] = 0;
    ierr = PetscOptionsGetString(PETSC_NULL,"-malloc_log",fname,250,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_log_threshold",&flg2);CHKERRQ(ierr);
    if (flg1 && fname[0]) {
      int  err;

      if (!rank) {
        fd = fopen(fname,"w");
        if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",fname);
      }
      ierr = PetscMallocDumpLog(fd);CHKERRQ(ierr);
      if (fd) {
        err = fclose(fd);
        if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      }
    } else if (flg1 || flg2) {
      ierr = PetscMallocDumpLog(stdout);CHKERRQ(ierr);
    }
  }
  /* Can be destroyed only after all the options are used */
  ierr = PetscOptionsDestroy();CHKERRQ(ierr);

  PetscGlobalArgc = 0;
  PetscGlobalArgs = 0;

#if defined(PETSC_USE_REAL___FLOAT128)
  ierr = MPI_Type_free(&MPIU___FLOAT128);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = MPI_Type_free(&MPIU___COMPLEX128);CHKERRQ(ierr);
#endif
  ierr = MPI_Op_free(&MPIU_MAX);CHKERRQ(ierr);
  ierr = MPI_Op_free(&MPIU_MIN);CHKERRQ(ierr);
#endif

#if defined(PETSC_USE_COMPLEX)
#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
  ierr = MPI_Type_free(&MPIU_C_DOUBLE_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_free(&MPIU_C_COMPLEX);CHKERRQ(ierr);
#endif
#endif

#if (defined(PETSC_USE_COMPLEX) && !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)) || defined(PETSC_USE_REAL___FLOAT128)
  ierr = MPI_Op_free(&MPIU_SUM);CHKERRQ(ierr);
#endif

  ierr = MPI_Type_free(&MPIU_2SCALAR);CHKERRQ(ierr);
  ierr = MPI_Type_free(&MPIU_2INT);CHKERRQ(ierr);
  ierr = MPI_Op_free(&PetscMaxSum_Op);CHKERRQ(ierr);
  ierr = MPI_Op_free(&PetscADMax_Op);CHKERRQ(ierr);
  ierr = MPI_Op_free(&PetscADMin_Op);CHKERRQ(ierr);

  /*
     Destroy any known inner MPI_Comm's and attributes pointing to them
     Note this will not destroy any new communicators the user has created.

     If all PETSc objects were not destroyed those left over objects will have hanging references to
     the MPI_Comms that were freed; but that is ok because those PETSc objects will never be used again
 */
  {
    PetscCommCounter *counter;
    PetscMPIInt      flg;
    MPI_Comm         icomm;
    void             *ptr;
    ierr  = MPI_Attr_get(PETSC_COMM_SELF,Petsc_InnerComm_keyval,&ptr,&flg);CHKERRQ(ierr);
    if (flg) {
      /*  Use PetscMemcpy() because casting from pointer to integer of different size is not allowed with some compilers  */
      ierr = PetscMemcpy(&icomm,&ptr,sizeof(MPI_Comm));CHKERRQ(ierr);
      ierr = MPI_Attr_get(icomm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      ierr = MPI_Attr_delete(PETSC_COMM_SELF,Petsc_InnerComm_keyval);CHKERRQ(ierr);
      ierr = MPI_Attr_delete(icomm,Petsc_Counter_keyval);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&icomm);CHKERRQ(ierr);
    }
    ierr  = MPI_Attr_get(PETSC_COMM_WORLD,Petsc_InnerComm_keyval,&ptr,&flg);CHKERRQ(ierr);
    if (flg) {
      /*  Use PetscMemcpy() because casting from pointer to integer of different size is not allowed with some compilers  */
      ierr = PetscMemcpy(&icomm,&ptr,sizeof(MPI_Comm));CHKERRQ(ierr);
      ierr = MPI_Attr_get(icomm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_CORRUPT,"Inner MPI_Comm does not have expected tag/name counter, problem with corrupted memory");

      ierr = MPI_Attr_delete(PETSC_COMM_WORLD,Petsc_InnerComm_keyval);CHKERRQ(ierr);
      ierr = MPI_Attr_delete(icomm,Petsc_Counter_keyval);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&icomm);CHKERRQ(ierr);
    }
  }

  ierr = MPI_Keyval_free(&Petsc_Counter_keyval);CHKERRQ(ierr);
  ierr = MPI_Keyval_free(&Petsc_InnerComm_keyval);CHKERRQ(ierr);
  ierr = MPI_Keyval_free(&Petsc_OuterComm_keyval);CHKERRQ(ierr);

#if defined(PETSC_HAVE_CUDA)
  {
    PetscInt p;
    for (p = 0; p < PetscGlobalSize; ++p) {
      if (p == PetscGlobalRank) {cublasShutdown();}
      ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    }
  }
#endif

  ierr = PetscInfo(0,"PETSc successfully ended!\n");CHKERRQ(ierr);
  if (PetscBeganMPI) {
#if defined(PETSC_HAVE_MPI_FINALIZED)
    PetscMPIInt flag;
    ierr = MPI_Finalized(&flag);CHKERRQ(ierr);
    if (flag) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"MPI_Finalize() has already been called, even though MPI_Init() was called by PetscInitialize()");
#endif
    ierr = MPI_Finalize();CHKERRQ(ierr);
  }
/*

     Note: In certain cases PETSC_COMM_WORLD is never MPI_Comm_free()ed because
   the communicator has some outstanding requests on it. Specifically if the
   flag PETSC_HAVE_BROKEN_REQUEST_FREE is set (for IBM MPI implementation). See
   src/vec/utils/vpscat.c. Due to this the memory allocated in PetscCommDuplicate()
   is never freed as it should be. Thus one may obtain messages of the form
   [ 1] 8 bytes PetscCommDuplicate() line 645 in src/sys/mpiu.c indicating the
   memory was not freed.

*/
  ierr = PetscMallocClear();CHKERRQ(ierr);
  PetscInitializeCalled = PETSC_FALSE;
  PetscFinalizeCalled   = PETSC_TRUE;
  PetscFunctionReturn(ierr);
}

#if defined(PETSC_MISSING_LAPACK_lsame_)
EXTERN_C_BEGIN
int lsame_(char *a,char *b)
{
  if (*a == *b) return 1;
  if (*a + 32 == *b) return 1;
  if (*a - 32 == *b) return 1;
  return 0;
}
EXTERN_C_END
#endif

#if defined(PETSC_MISSING_LAPACK_lsame)
EXTERN_C_BEGIN
int lsame(char *a,char *b)
{
  if (*a == *b) return 1;
  if (*a + 32 == *b) return 1;
  if (*a - 32 == *b) return 1;
  return 0;
}
EXTERN_C_END
#endif
