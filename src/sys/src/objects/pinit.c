#define PETSC_DLL
/*
   This file defines the initialization of PETSc, including PetscInitialize()
*/

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"

EXTERN PetscErrorCode PetscLogBegin_Private(void);

/* -----------------------------------------------------------------------------------------*/

extern FILE *petsc_history;

EXTERN PetscErrorCode PetscInitialize_DynamicLibraries(void);
EXTERN PetscErrorCode PetscFinalize_DynamicLibraries(void);
EXTERN PetscErrorCode PetscFListDestroyAll(void);
EXTERN PetscErrorCode PetscSequentialPhaseBegin_Private(MPI_Comm,int);
EXTERN PetscErrorCode PetscSequentialPhaseEnd_Private(MPI_Comm,int);
EXTERN PetscErrorCode PetscLogCloseHistoryFile(FILE **);

/* this is used by the _, __, and ___ macros (see include/petscerror.h) */
PetscErrorCode __gierr = 0;

/* user may set this BEFORE calling PetscInitialize() */
MPI_Comm PETSC_COMM_WORLD = 0;

/*
     Declare and set all the string names of the PETSc enums
*/
const char *PetscTruths[]    = {"FALSE","TRUE","PetscTruth","PETSC_",0};
const char *PetscDataTypes[] = {"INT", "DOUBLE", "COMPLEX",
                                "LONG","SHORT",  "FLOAT",
                                "CHAR","LOGICAL","ENUM","TRUTH","PetscDataType","PETSC_",0};

/*
       Checks the options database for initializations related to the 
    PETSc components
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCheckInitial_Components"
PetscErrorCode PETSC_DLLEXPORT PetscOptionsCheckInitial_Components(void)
{
  MPI_Comm   comm = PETSC_COMM_WORLD;
  PetscTruth flg1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {
#if defined (PETSC_USE_LOG)
    ierr = (*PetscHelpPrintf)(comm,"------Additional PETSc component options--------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_summary_exclude: <vec,mat,pc.ksp,snes>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_info_exclude: <null,vec,mat,pc,ksp,snes,ts>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");CHKERRQ(ierr);
#endif
  }
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
PetscErrorCode PETSC_DLLEXPORT PetscInitializeNoArguments(void)
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
  
   Level: beginner

.seealso: PetscInitialize(), PetscInitializeNoArguments(), PetscInitializeFortran()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscInitialized(PetscTruth *isInitialized)
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
PetscErrorCode PETSC_DLLEXPORT PetscFinalized(PetscTruth *isFinalized)
{
  PetscFunctionBegin;
  PetscValidPointer(isFinalized, 1);
  *isFinalized = PetscFinalizeCalled;
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode        PetscOptionsCheckInitial_Private(void);
extern PetscTruth PetscBeganMPI;

/*
       This function is the MPI reduction operation used to compute the sum of the 
   first half of the datatype and the max of the second half.
*/
MPI_Op PetscMaxSum_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscMaxSum_Local"
void PETSC_DLLEXPORT PetscMaxSum_Local(void *in,void *out,int *cnt,MPI_Datatype *datatype)
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
  PetscStackPop;
  return;
}
EXTERN_C_END

/*
    Returns the max of the first entry owned by this processor and the
sum of the second entry.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscMaxSum"
PetscErrorCode PETSC_DLLEXPORT PetscMaxSum(MPI_Comm comm,const PetscInt nprocs[],PetscInt *max,PetscInt *sum)
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
MPI_Op PETSC_DLLEXPORT PetscADMax_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscADMax_Local"
void PETSC_DLLEXPORT PetscADMax_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
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

  PetscStackPop;
  return;
}
EXTERN_C_END

MPI_Op PETSC_DLLEXPORT PetscADMin_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscADMin_Local"
void PETSC_DLLEXPORT PetscADMin_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
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

  PetscStackPop;
  return;
}
EXTERN_C_END
/* ---------------------------------------------------------------------------------------*/

#if defined(PETSC_USE_COMPLEX)
MPI_Op PetscSum_Op = 0;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscSum_Local"
void PETSC_DLLEXPORT PetscSum_Local(void *in,void *out,PetscMPIInt *cnt,MPI_Datatype *datatype)
{
  PetscScalar *xin = (PetscScalar *)in,*xout = (PetscScalar*)out;
  PetscInt         i,count = *cnt;

  PetscFunctionBegin;
  if (*datatype != MPIU_SCALAR) {
    (*PetscErrorPrintf)("Can only handle MPIU_SCALAR data (i.e. double or complex) types");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  for (i=0; i<count; i++) {
    xout[i] += xin[i]; 
  }

  PetscStackPop;
  return;
}
EXTERN_C_END
#endif

static int  PetscGlobalArgc   = 0;
static char **PetscGlobalArgs = 0;

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

   Concepts: command line arguments
   
.seealso: PetscFinalize(), PetscInitializeFortran()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetArgs(int *argc,char ***args)
{
  PetscFunctionBegin;
  if (!PetscGlobalArgs) {
    SETERRQ(PETSC_ERR_ORDER,"You must call after PetscInitialize() but before PetscFinalize()");
  }
  *argc = PetscGlobalArgc;
  *args = PetscGlobalArgs;
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
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use PETSC_NULL for default)
-  help - [optional] Help message to print, use PETSC_NULL for no message

   If you wish PETSc to run on a subcommunicator of MPI_COMM_WORLD, create that
   communicator first and assign it to PETSC_COMM_WORLD BEFORE calling PetscInitialize()

   Options Database Keys:
+  -start_in_debugger [noxterm,dbx,xdb,gdb,...] - Starts program in debugger
.  -on_error_attach_debugger [noxterm,dbx,xdb,gdb,...] - Starts debugger when error detected
.  -on_error_emacs <machinename> causes emacsclient to jump to error file
.  -debugger_nodes [node1,node2,...] - Indicates nodes to start in debugger
.  -debugger_pause [sleeptime] (in seconds) - Pauses debugger
.  -stop_for_debugger - Print message on how to attach debugger manually to 
                        process and wait (-debugger_pause) seconds for attachment
.  -malloc - Indicates use of PETSc error-checking malloc
.  -malloc no - Indicates not to use error-checking malloc
.  -fp_trap - Stops on floating point exceptions (Note that on the
              IBM RS6000 this slows code by at least a factor of 10.)
.  -no_signal_handler - Indicates not to trap error signals
.  -shared_tmp - indicates /tmp directory is shared by all processors
.  -not_shared_tmp - each processor has own /tmp
.  -tmp - alternative name of /tmp directory
.  -get_total_flops - returns total flops done by all processors
-  -memory_info - Print memory usage at end of run

   Options Database Keys for Profiling:
   See the Profiling chapter of the users manual for details.
+  -log_trace [filename] - Print traces of all PETSc calls
        to the screen (useful to determine where a program
        hangs without running in the debugger).  See PetscLogTraceBegin().
.  -log_info <optional filename> - Prints verbose information to the screen
-  -log_info_exclude <null,vec,mat,pc,ksp,snes,ts> - Excludes some of the verbose messages

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
-   file - [optional] PETSc database file name, defaults to 
           ~username/.petscrc (use PETSC_NULL_CHARACTER for default)
           
   Important Fortran Note:
   In Fortran, you MUST use PETSC_NULL_CHARACTER to indicate a
   null character string; you CANNOT just use PETSC_NULL as 
   in the C version.  See the users manual for details.


   Concepts: initializing PETSc
   
.seealso: PetscFinalize(), PetscInitializeFortran(), PetescGetArgs()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscErrorCode ierr;
  PetscMPIInt    flag, size;
  PetscTruth     flg;
  char           hostname[256];

  PetscFunctionBegin;
  if (PetscInitializeCalled) PetscFunctionReturn(0);

  /* this must be initialized in a routine, not as a constant declaration*/
  PETSC_STDOUT = stdout;  

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
    if (PETSC_COMM_WORLD) SETERRQ(PETSC_ERR_SUP,"You cannot set PETSC_COMM_WORLD if you have not initialized MPI first");
    ierr          = MPI_Init(argc,args);CHKERRQ(ierr);
    PetscBeganMPI = PETSC_TRUE;
  }
  if (argc && args) {
    PetscGlobalArgc = *argc;
    PetscGlobalArgs = *args;
  }
  PetscInitializeCalled = PETSC_TRUE;
  PetscFinalizeCalled   = PETSC_FALSE;

  if (!PETSC_COMM_WORLD) {
    PETSC_COMM_WORLD = MPI_COMM_WORLD;
  }

  /* Done after init due to a bug in MPICH-GM? */
  ierr = PetscErrorPrintfInitialize();CHKERRQ(ierr);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&PetscGlobalRank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&PetscGlobalSize);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  /* 
     Initialized the global complex variable; this is because with 
     shared libraries the constructors for global variables
     are not called; at least on IRIX.
  */
  {
    PetscScalar ic(0.0,1.0);
    PETSC_i = ic; 
  }
  ierr = MPI_Type_contiguous(2,MPIU_REAL,&MPIU_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_COMPLEX);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscSum_Local,1,&PetscSum_Op);CHKERRQ(ierr);
#endif

  /*
     Create the PETSc MPI reduction operator that sums of the first
     half of the entries and maxes the second half.
  */
  ierr = MPI_Op_create(PetscMaxSum_Local,1,&PetscMaxSum_Op);CHKERRQ(ierr);

  ierr = MPI_Type_contiguous(2,MPIU_SCALAR,&MPIU_2SCALAR);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_2SCALAR);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscADMax_Local,1,&PetscADMax_Op);CHKERRQ(ierr);
  ierr = MPI_Op_create(PetscADMin_Local,1,&PetscADMin_Op);CHKERRQ(ierr);

  ierr = MPI_Type_contiguous(2,MPIU_INT,&MPIU_2INT);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&MPIU_2INT);CHKERRQ(ierr);

  /*
     Build the options database and check for user setup requests
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
  ierr = PetscLogBegin_Private();CHKERRQ(ierr);

  /*
     Load the dynamic libraries (on machines that support them), this registers all
     the solvers etc. (On non-dynamic machines this initializes the PetscDraw and PetscViewer classes)
  */
  ierr = PetscInitialize_DynamicLibraries();CHKERRQ(ierr);

  /*
     Initialize all the default viewers
  */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscLogInfo((0,"PetscInitialize:PETSc successfully started: number of processors = %d\n",size));CHKERRQ(ierr);
  ierr = PetscGetHostName(hostname,256);CHKERRQ(ierr);
  ierr = PetscLogInfo((0,"PetscInitialize:Running on machine: %s\n",hostname));CHKERRQ(ierr);

  ierr = PetscOptionsCheckInitial_Components();CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscFinalize"
/*@C 
   PetscFinalize - Checks for options to be called at the conclusion
   of the program. MPI_Finalize() is called only if the user had not
   called MPI_Init() before calling PetscInitialize().

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -options_table - Calls PetscOptionsPrint()
.  -options_left - Prints unused options that remain in the database
.  -options_left no - Does not print unused options that remain in the database
.  -mpidump - Calls PetscMPIDump()
.  -malloc_dump - Calls PetscMallocDump()
.  -malloc_info - Prints total memory usage
.  -malloc_debug - Calls PetscMallocDebug(), checks allocated memory for corruption while running
-  -malloc_log - Prints summary of memory usage

   Options Database Keys for Profiling:
   See the Profiling chapter of the users manual for details.
+  -log_summary [filename] - Prints summary of flop and timing
        information to screen. If the filename is specified the
        summary is written to the file. (for code compiled with 
        PETSC_USE_LOG).  See PetscLogPrintSummary().
.  -log_all [filename] - Logs extensive profiling information
        (for code compiled with PETSC_USE_LOG). See PetscLogDump(). 
.  -log [filename] - Logs basic profiline information (for
        code compiled with PETSC_USE_LOG).  See PetscLogDump().
.  -log_sync - Log the synchronization in scatters, inner products
        and norms
-  -log_mpe [filename] - Creates a logfile viewable by the 
      utility Upshot/Nupshot (in MPICH distribution)

   Level: beginner

   Note:
   See PetscInitialize() for more general runtime options.

.seealso: PetscInitialize(), PetscOptionsPrint(), PetscMallocDump(), PetscMPIDump(), PetscEnd()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscFinalize(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  int            nopt;
  PetscTruth     flg1,flg2,flg3;
  
  PetscFunctionBegin;

  if (!PetscInitializeCalled) {
    (*PetscErrorPrintf)("PetscInitialize() must be called before PetscFinalize()\n");
    PetscFunctionReturn(0);
  }

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_info",&flg2);CHKERRQ(ierr);
  if (!flg2) {
    ierr = PetscOptionsHasName(PETSC_NULL,"-memory_info",&flg2);CHKERRQ(ierr);
  }
  if (flg2) {
    ierr = PetscMemoryShowUsage(PETSC_VIEWER_STDOUT_WORLD,"Summary of Memory Usage in PETSc\n");CHKERRQ(ierr);
  }

  /* Destroy auxiliary packages */
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscViewerMathematicaFinalizePackage();CHKERRQ(ierr);
#endif
  ierr = PetscPLAPACKFinalizePackage();CHKERRQ(ierr);

  /*
     Destroy all the function registration lists created
  */
  ierr = PetscFinalize_DynamicLibraries();CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscOptionsHasName(PETSC_NULL,"-get_total_flops",&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscLogDouble flops = 0;
    ierr = MPI_Reduce(&_TotalFlops,&flops,1,MPI_DOUBLE,MPI_SUM,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Total flops over all processors %g\n",flops);CHKERRQ(ierr);
  }
#endif

  /*
     Free all objects registered with PetscObjectRegisterDestroy() such ast
    PETSC_VIEWER_XXX_().
  */
  ierr = PetscObjectRegisterDestroyAll();CHKERRQ(ierr);  

#if defined(PETSC_USE_DEBUG)
  if (PetscStackActive) {
    ierr = PetscStackDestroy();CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_USE_LOG)
  {
    char mname[PETSC_MAX_PATH_LEN];
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
      if (mname[0])  {ierr = PetscLogPrintSummary(PETSC_COMM_WORLD,mname);CHKERRQ(ierr);}
      else           {ierr = PetscLogPrintSummary(PETSC_COMM_WORLD,0);CHKERRQ(ierr);}
    }

    mname[0] = 0;
    ierr = PetscOptionsGetString(PETSC_NULL,"-log_all",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(PETSC_NULL,"-log",mname,PETSC_MAX_PATH_LEN,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2){
      if (mname[0]) PetscLogDump(mname); 
      else          PetscLogDump(0);
    }
    ierr = PetscLogDestroy();CHKERRQ(ierr);
  }
#endif
  ierr = PetscOptionsHasName(PETSC_NULL,"-no_signal_handler",&flg1);CHKERRQ(ierr);
  if (!flg1) { ierr = PetscPopSignalHandler();CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL,"-mpidump",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscMPIDump(stdout);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_dump",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-options_table",&flg2);CHKERRQ(ierr);
  if (flg2) {
    if (!rank) {ierr = PetscOptionsPrint(stdout);CHKERRQ(ierr);}
  }

  /* to prevent PETSc -options_left from warning */
  ierr = PetscOptionsHasName(PETSC_NULL,"-nox_warning",&flg1);CHKERRQ(ierr)
  ierr = PetscOptionsHasName(PETSC_NULL,"-error_output_stderr",&flg1);CHKERRQ(ierr);

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-options_left",&flg2,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsAllUsed(&nopt);CHKERRQ(ierr);
  if (flg2) {
    ierr = PetscOptionsPrint(stdout);CHKERRQ(ierr);
    if (!nopt) { 
      ierr = PetscPrintf(PETSC_COMM_WORLD,"There are no unused options.\n");CHKERRQ(ierr);
    } else if (nopt == 1) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"There is one unused database option. It is:\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"There are %d unused database options. They are:\n",nopt);CHKERRQ(ierr);
    }
  } 
#if defined(PETSC_USE_DEBUG)
  if (nopt && !flg1 && !flg2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! There are options you set that were not used!\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING! could be spelling mistake, etc!\n");CHKERRQ(ierr);
    ierr = PetscOptionsLeft();CHKERRQ(ierr);
  } else if (nopt && flg2) {
#else 
  if (nopt && flg2) {
#endif
    ierr = PetscOptionsLeft();CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-log_history",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscLogCloseHistoryFile(&petsc_history);CHKERRQ(ierr);
    petsc_history = 0;
  }

  ierr = PetscLogInfoAllow(PETSC_FALSE,PETSC_NULL);CHKERRQ(ierr);

  /*
       Free all the registered create functions, such as KSPList, VecList, SNESList, etc
  */
  ierr = PetscFListDestroyAll();CHKERRQ(ierr); 

  ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_dump",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_log",&flg3);CHKERRQ(ierr);
  if (flg1) {
    char fname[PETSC_MAX_PATH_LEN];
    FILE *fd;
    
    fname[0] = 0;
    ierr = PetscOptionsGetString(PETSC_NULL,"-malloc_dump",fname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && fname[0]) {
      char sname[PETSC_MAX_PATH_LEN];

      sprintf(sname,"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); if (!fd) SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",sname);
      ierr = PetscMallocDump(fd);CHKERRQ(ierr);
      fclose(fd);
    } else {
      MPI_Comm local_comm;

      ierr = MPI_Comm_dup(MPI_COMM_WORLD,&local_comm);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseBegin_Private(local_comm,1);CHKERRQ(ierr);
        ierr = PetscMallocDump(stdout);CHKERRQ(ierr);
      ierr = PetscSequentialPhaseEnd_Private(local_comm,1);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&local_comm);CHKERRQ(ierr);
    }
  }
  if (flg3) {
    char fname[PETSC_MAX_PATH_LEN];
    FILE *fd;
    
    fname[0] = 0;
    ierr = PetscOptionsGetString(PETSC_NULL,"-malloc_log",fname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && fname[0]) {
      char sname[PETSC_MAX_PATH_LEN];

      sprintf(sname,"%s_%d",fname,rank);
      fd   = fopen(sname,"w"); if (!fd) SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open log file: %s",sname);
      ierr = PetscMallocDumpLog(fd);CHKERRQ(ierr); 
      fclose(fd);
    } else {
      ierr = PetscMallocDumpLog(stdout);CHKERRQ(ierr); 
    }
  }
  /* Can be destroyed only after all the options are used */
  ierr = PetscOptionsDestroy();CHKERRQ(ierr);

  PetscGlobalArgc = 0;
  PetscGlobalArgs = 0;

#if defined(PETSC_USE_COMPLEX)
  ierr = MPI_Op_free(&PetscSum_Op);CHKERRQ(ierr);
  ierr = MPI_Type_free(&MPIU_COMPLEX);CHKERRQ(ierr);
#endif
  ierr = MPI_Type_free(&MPIU_2SCALAR);CHKERRQ(ierr);
  ierr = MPI_Type_free(&MPIU_2INT);CHKERRQ(ierr);
  ierr = MPI_Op_free(&PetscMaxSum_Op);CHKERRQ(ierr);
  ierr = MPI_Op_free(&PetscADMax_Op);CHKERRQ(ierr);
  ierr = MPI_Op_free(&PetscADMin_Op);CHKERRQ(ierr);

  ierr = PetscLogInfo((0,"PetscFinalize:PETSc successfully ended!\n"));CHKERRQ(ierr);
  if (PetscBeganMPI) {
    ierr = MPI_Finalize();CHKERRQ(ierr);
  }

/*

     Note: In certain cases PETSC_COMM_WORLD is never MPI_Comm_free()ed because 
   the communicator has some outstanding requests on it. Specifically if the 
   flag PETSC_HAVE_BROKEN_REQUEST_FREE is set (for IBM MPI implementation). See 
   src/vec/utils/vpscat.c. Due to this the memory allocated in PetscCommDuplicate()
   is never freed as it should be. Thus one may obtain messages of the form
   [ 1] 8 bytes PetscCommDuplicate() line 645 in src/sys/src/mpiu.c indicating the
   memory was not freed.

*/
  ierr = PetscClearMalloc();CHKERRQ(ierr);
  PetscInitializeCalled = PETSC_FALSE;
  PetscFinalizeCalled   = PETSC_TRUE;
  PetscFunctionReturn(ierr);
}

/*
     These may be used in code that ADIC is to be used on
*/

#undef __FUNCT__  
#define __FUNCT__ "PetscGlobalMax"
/*@C
      PetscGlobalMax - Computes the maximum value over several processors

     Collective on MPI_Comm

   Input Parameters:
+   local - the local value
-   comm - the processors that find the maximum

   Output Parameter:
.   result - the maximum value
  
   Level: intermediate

   Notes:
     These functions are to be used inside user functions that are to be processed with 
   ADIC. PETSc will automatically provide differentiated versions of these functions

.seealso: PetscGlobalMin(), PetscGlobalSum()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGlobalMax(PetscReal* local,PetscReal* result,MPI_Comm comm)
{
  return MPI_Allreduce(local,result,1,MPIU_REAL,MPI_MAX,comm);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGlobalMin"
/*@C
      PetscGlobalMin - Computes the minimum value over several processors

     Collective on MPI_Comm

   Input Parameters:
+   local - the local value
-   comm - the processors that find the minimum

   Output Parameter:
.   result - the minimum value
  
   Level: intermediate

   Notes:
     These functions are to be used inside user functions that are to be processed with 
   ADIC. PETSc will automatically provide differentiated versions of these functions

.seealso: PetscGlobalMax(), PetscGlobalSum()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGlobalMin(PetscReal* local,PetscReal* result,MPI_Comm comm)
{
  return MPI_Allreduce(local,result,1,MPIU_REAL,MPI_MIN,comm);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGlobalSum"
/*@C
      PetscGlobalSum - Computes the sum over sever processors

     Collective on MPI_Comm

   Input Parameters:
+   local - the local value
-   comm - the processors that find the sum

   Output Parameter:
.   result - the sum
  
   Level: intermediate

   Notes:
     These functions are to be used inside user functions that are to be processed with 
   ADIC. PETSc will automatically provide differentiated versions of these functions

.seealso: PetscGlobalMin(), PetscGlobalMax()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGlobalSum(PetscScalar* local,PetscScalar* result,MPI_Comm comm)
{
  return MPI_Allreduce(local,result,1,MPIU_SCALAR,PetscSum_Op,comm);
}


