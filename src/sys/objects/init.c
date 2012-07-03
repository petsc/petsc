/*

   This file defines part of the initialization of PETSc

  This file uses regular malloc and free because it cannot known 
  what malloc is being used until it has already processed the input.
*/

#include <petscsys.h>        /*I  "petscsys.h"   I*/

#if defined(PETSC_HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#if defined(PETSC_HAVE_VALGRIND)
#  include <valgrind/valgrind.h>
#  define PETSC_RUNNING_ON_VALGRIND RUNNING_ON_VALGRIND
#else
#  define PETSC_RUNNING_ON_VALGRIND PETSC_FALSE
#endif

/* ------------------------Nasty global variables -------------------------------*/
/*
     Indicates if PETSc started up MPI, or it was 
   already started before PETSc was initialized.
*/
PetscBool    PetscBeganMPI         = PETSC_FALSE;
PetscBool    PetscInitializeCalled = PETSC_FALSE;
PetscBool    PetscFinalizeCalled   = PETSC_FALSE;
PetscMPIInt  PetscGlobalRank = -1;
PetscMPIInt  PetscGlobalSize = -1;

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_COMPLEX_INSTANTIATE)
template <> class std::complex<double>; /* instantiate complex template class */
#endif
#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
MPI_Datatype   MPIU_C_DOUBLE_COMPLEX;
MPI_Datatype   MPIU_C_COMPLEX;
#endif
PetscScalar    PETSC_i;
#else
PetscScalar    PETSC_i = 0.0;
#endif
#if defined(PETSC_USE_REAL___FLOAT128)
MPI_Datatype   MPIU___FLOAT128 = 0;
#endif
MPI_Datatype   MPIU_2SCALAR = 0;
MPI_Datatype   MPIU_2INT = 0;

/*
       Function that is called to display all error messages
*/
PetscErrorCode  (*PetscErrorPrintf)(const char [],...)          = PetscErrorPrintfDefault;
PetscErrorCode  (*PetscHelpPrintf)(MPI_Comm,const char [],...)  = PetscHelpPrintfDefault;
#if defined(PETSC_HAVE_MATLAB_ENGINE)
PetscErrorCode  (*PetscVFPrintf)(FILE*,const char[],va_list)    = PetscVFPrintf_Matlab;
#else
PetscErrorCode  (*PetscVFPrintf)(FILE*,const char[],va_list)    = PetscVFPrintfDefault;
#endif
/*
  This is needed to turn on/off cusp synchronization 
*/
PetscBool   PetscCUSPSynchronize = PETSC_FALSE;

/* pthread_key for PetscStack */
#if defined(PETSC_HAVE_PTHREADCLASSES) && !defined(PETSC_PTHREAD_LOCAL)
pthread_key_t petscstack_key;
#endif

/* ------------------------------------------------------------------------------*/
/* 
   Optional file where all PETSc output from various prints is saved
*/
FILE *petsc_history = PETSC_NULL;

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenHistoryFile"
PetscErrorCode  PetscOpenHistoryFile(const char filename[],FILE **fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  char           pfile[PETSC_MAX_PATH_LEN],pname[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN],date[64];
  char           version[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    char        arch[10];
    int         err;
    PetscViewer viewer;

    ierr = PetscGetArchType(arch,10);CHKERRQ(ierr);
    ierr = PetscGetDate(date,64);CHKERRQ(ierr);
    ierr = PetscGetVersion(version,256);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    if (filename) {
      ierr = PetscFixFilename(filename,fname);CHKERRQ(ierr);
    } else {
      ierr = PetscGetHomeDirectory(pfile,240);CHKERRQ(ierr);
      ierr = PetscStrcat(pfile,"/.petschistory");CHKERRQ(ierr);
      ierr = PetscFixFilename(pfile,fname);CHKERRQ(ierr);
    }

    *fd = fopen(fname,"a"); if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file: %s",fname);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"---------------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"%s %s\n",version,date);CHKERRQ(ierr);
    ierr = PetscGetProgramName(pname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"%s on a %s, %d proc. with options:\n",pname,arch,size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpenWithFILE(PETSC_COMM_WORLD,*fd,&viewer);CHKERRQ(ierr);
    ierr = PetscOptionsView(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"---------------------------------------------------------\n");CHKERRQ(ierr);
    err = fflush(*fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");        
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "PetscCloseHistoryFile"
PetscErrorCode  PetscCloseHistoryFile(FILE **fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           date[64];
  int            err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscGetDate(date,64);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"---------------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"Finished at %s\n",date);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"---------------------------------------------------------\n");CHKERRQ(ierr);
    err = fflush(*fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");        
    err = fclose(*fd);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");        
  }
  PetscFunctionReturn(0); 
}

/* ------------------------------------------------------------------------------*/

/* 
   This is ugly and probably belongs somewhere else, but I want to 
  be able to put a true MPI abort error handler with command line args.

    This is so MPI errors in the debugger will leave all the stack 
  frames. The default MP_Abort() cleans up and exits thus providing no useful information
  in the debugger hence we call abort() instead of MPI_Abort().
*/

#undef __FUNCT__  
#define __FUNCT__ "Petsc_MPI_AbortOnError"
void Petsc_MPI_AbortOnError(MPI_Comm *comm,PetscMPIInt *flag) 
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("MPI error %d\n",*flag);
  abort();
}

#undef __FUNCT__  
#define __FUNCT__ "Petsc_MPI_DebuggerOnError"
void Petsc_MPI_DebuggerOnError(MPI_Comm *comm,PetscMPIInt *flag) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  (*PetscErrorPrintf)("MPI error %d\n",*flag);
  ierr = PetscAttachDebugger();
  if (ierr) { /* hopeless so get out */
    MPI_Abort(*comm,*flag);
  }
}

#undef __FUNCT__  
#define __FUNCT__ "PetscEnd"
/*@C 
   PetscEnd - Calls PetscFinalize() and then ends the program. This is useful if one 
     wishes a clean exit somewhere deep in the program.

   Collective on PETSC_COMM_WORLD

   Options Database Keys are the same as for PetscFinalize()

   Level: advanced

   Note:
   See PetscInitialize() for more general runtime options.

.seealso: PetscInitialize(), PetscOptionsView(), PetscMallocDump(), PetscMPIDump(), PetscFinalize()
@*/
PetscErrorCode  PetscEnd(void)
{
  PetscFunctionBegin;
  PetscFinalize();
  exit(0);
  return 0;
}

PetscBool    PetscOptionsPublish = PETSC_FALSE;
extern PetscErrorCode        PetscSetUseTrMalloc_Private(void);
extern PetscBool  petscsetmallocvisited;
static char       emacsmachinename[256];

PetscErrorCode (*PetscExternalVersionFunction)(MPI_Comm) = 0;
PetscErrorCode (*PetscExternalHelpFunction)(MPI_Comm)    = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscSetHelpVersionFunctions"
/*@C 
   PetscSetHelpVersionFunctions - Sets functions that print help and version information
   before the PETSc help and version information is printed. Must call BEFORE PetscInitialize().
   This routine enables a "higher-level" package that uses PETSc to print its messages first.

   Input Parameter:
+  help - the help function (may be PETSC_NULL)
-  version - the version function (may be PETSC_NULL)

   Level: developer

   Concepts: package help message

@*/
PetscErrorCode  PetscSetHelpVersionFunctions(PetscErrorCode (*help)(MPI_Comm),PetscErrorCode (*version)(MPI_Comm))
{
  PetscFunctionBegin;
  PetscExternalHelpFunction    = help;
  PetscExternalVersionFunction = version;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PTHREADCLASSES)
extern PetscErrorCode PetscOptionsCheckInitial_Private_Pthread(void);
#endif

#if defined(PETSC_HAVE_CUDA)
#include <cublas.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCheckInitial_Private"
PetscErrorCode  PetscOptionsCheckInitial_Private(void)
{
  char           string[64],mname[PETSC_MAX_PATH_LEN],*f;
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE,flg4 = PETSC_FALSE,flag;
  PetscErrorCode ierr;
  PetscReal      si;
  int            i;
  PetscMPIInt    rank;
  char           version[256];
#if defined(PETSC_USE_SERVER)
  PetscBool      flgz;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /*
      Setup the memory management; support for tracing malloc() usage 
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_log",&flg3);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc",&flg1,&flg2);CHKERRQ(ierr);
  if ((!flg2 || flg1) && !petscsetmallocvisited) {
    if (flg2 || !(PETSC_RUNNING_ON_VALGRIND)) {
      /* turn off default -malloc if valgrind is being used */
      ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr); 
    }
  }
#else
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc_dump",&flg1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc",&flg2,PETSC_NULL);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3) {ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr);}
#endif
  if (flg3) {
    ierr = PetscMallocSetDumpLog();CHKERRQ(ierr); 
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc_debug",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) { 
    ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr);
    ierr = PetscMallocDebug(PETSC_TRUE);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc_test",&flg1,PETSC_NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  if (flg1 && !PETSC_RUNNING_ON_VALGRIND) {
    ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr);
    ierr = PetscMallocSetDumpLog();CHKERRQ(ierr);
    ierr = PetscMallocDebug(PETSC_TRUE);CHKERRQ(ierr);
  }
#endif

  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc_info",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (!flg1) {
    flg1 = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL,"-memory_info",&flg1,PETSC_NULL);CHKERRQ(ierr);
  }
  if (flg1) {
    ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);
  }

  /*
      Set the display variable for graphics
  */
  ierr = PetscSetDisplay();CHKERRQ(ierr);

  /*
      Print the PETSc version information
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-v",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-version",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg3);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3){

    /*
       Print "higher-level" package version message 
    */
    if (PetscExternalVersionFunction) {
      ierr = (*PetscExternalVersionFunction)(comm);CHKERRQ(ierr);
    }

    ierr = PetscGetVersion(version,256);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"%s\n",version);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"%s",PETSC_AUTHOR_INFO);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/changes/index.html for recent updates.\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/faq.html for problems.\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/manualpages/index.html for help. \n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"Libraries linked from %s\n",PETSC_LIB_DIR);CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"--------------------------------------------\
------------------------------\n");CHKERRQ(ierr);
  }

  /*
       Print "higher-level" package help message 
  */
  if (flg3){
    if (PetscExternalHelpFunction) {
      ierr = (*PetscExternalHelpFunction)(comm);CHKERRQ(ierr);
    }
  }

  /*
      Setup the error handling
  */
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-on_error_abort",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) {
    ierr = MPI_Errhandler_set(PETSC_COMM_WORLD,MPI_ERRORS_ARE_FATAL);CHKERRQ(ierr);
    ierr = PetscPushErrorHandler(PetscAbortErrorHandler,0);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-on_error_mpiabort",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) { ierr = PetscPushErrorHandler(PetscMPIAbortErrorHandler,0);CHKERRQ(ierr);}
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mpi_return_on_error",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) {
    ierr = MPI_Errhandler_set(comm,MPI_ERRORS_RETURN);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-no_signal_handler",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (!flg1) {ierr = PetscPushSignalHandler(PetscDefaultSignalHandler,(void*)0);CHKERRQ(ierr);}
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-fp_trap",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) {ierr = PetscSetFPTrap(PETSC_FP_TRAP_ON);CHKERRQ(ierr);}

  /*
      Setup debugger information
  */
  ierr = PetscSetDefaultDebugger();CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-on_error_attach_debugger",string,64,&flg1);CHKERRQ(ierr);
  if (flg1) {
    MPI_Errhandler err_handler;

    ierr = PetscSetDebuggerFromString(string);CHKERRQ(ierr);
    ierr = MPI_Errhandler_create((MPI_Handler_function*)Petsc_MPI_DebuggerOnError,&err_handler);CHKERRQ(ierr);
    ierr = MPI_Errhandler_set(comm,err_handler);CHKERRQ(ierr);
    ierr = PetscPushErrorHandler(PetscAttachDebuggerErrorHandler,0);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(PETSC_NULL,"-debug_terminal",string,64,&flg1);CHKERRQ(ierr);
  if (flg1) { ierr = PetscSetDebugTerminal(string);CHKERRQ(ierr); }
  ierr = PetscOptionsGetString(PETSC_NULL,"-start_in_debugger",string,64,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-stop_for_debugger",string,64,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    PetscMPIInt    size;
    PetscInt       lsize,*nodes;
    MPI_Errhandler err_handler;
    /*
       we have to make sure that all processors have opened 
       connections to all other processors, otherwise once the 
       debugger has stated it is likely to receive a SIGUSR1
       and kill the program. 
    */
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    if (size > 2) {
      PetscMPIInt dummy = 0;
      MPI_Status  status;
      for (i=0; i<size; i++) {
        if (rank != i) {
          ierr = MPI_Send(&dummy,1,MPI_INT,i,109,PETSC_COMM_WORLD);CHKERRQ(ierr);
        }
      }
      for (i=0; i<size; i++) {
        if (rank != i) {
          ierr = MPI_Recv(&dummy,1,MPI_INT,i,109,PETSC_COMM_WORLD,&status);CHKERRQ(ierr);
        }
      }
    }
    /* check if this processor node should be in debugger */
    ierr  = PetscMalloc(size*sizeof(PetscInt),&nodes);CHKERRQ(ierr);
    lsize = size;
    ierr  = PetscOptionsGetIntArray(PETSC_NULL,"-debugger_nodes",nodes,&lsize,&flag);CHKERRQ(ierr);
    if (flag) {
      for (i=0; i<lsize; i++) {
        if (nodes[i] == rank) { flag = PETSC_FALSE; break; }
      }
    }
    if (!flag) {
      ierr = PetscSetDebuggerFromString(string);CHKERRQ(ierr);
      ierr = PetscPushErrorHandler(PetscAbortErrorHandler,0);CHKERRQ(ierr);
      if (flg1) {
        ierr = PetscAttachDebugger();CHKERRQ(ierr);
      } else {
        ierr = PetscStopForDebugger();CHKERRQ(ierr);
      }
      ierr = MPI_Errhandler_create((MPI_Handler_function*)Petsc_MPI_AbortOnError,&err_handler);CHKERRQ(ierr);
      ierr = MPI_Errhandler_set(comm,err_handler);CHKERRQ(ierr);
    }
    ierr = PetscFree(nodes);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-on_error_emacs",emacsmachinename,128,&flg1);CHKERRQ(ierr);
  if (flg1 && !rank) {ierr = PetscPushErrorHandler(PetscEmacsClientErrorHandler,emacsmachinename);CHKERRQ(ierr);}

#if defined(PETSC_USE_SERVER)
  ierr = PetscOptionsHasName(PETSC_NULL,"-server", &flgz);CHKERRQ(ierr);
  if (flgz){
    PetscInt port = PETSC_DECIDE;
    ierr = PetscOptionsGetInt(PETSC_NULL,"-server",&port,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscWebServe(PETSC_COMM_WORLD,(int)port);CHKERRQ(ierr);
  }
#endif

  /*
        Setup profiling and logging
  */
#if defined (PETSC_USE_INFO)
  { 
    char logname[PETSC_MAX_PATH_LEN]; logname[0] = 0;
    ierr = PetscOptionsGetString(PETSC_NULL,"-info",logname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && logname[0]) {
      ierr = PetscInfoAllow(PETSC_TRUE,logname);CHKERRQ(ierr);
    } else if (flg1) {
      ierr = PetscInfoAllow(PETSC_TRUE,PETSC_NULL);CHKERRQ(ierr); 
    }
  }
#endif
#if defined(PETSC_USE_LOG)
  mname[0] = 0;
  ierr = PetscOptionsGetString(PETSC_NULL,"-history",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1) {
    if (mname[0]) {
      ierr = PetscOpenHistoryFile(mname,&petsc_history);CHKERRQ(ierr);
    } else {
      ierr = PetscOpenHistoryFile(0,&petsc_history);CHKERRQ(ierr);
    }
  }
#if defined(PETSC_HAVE_MPE)
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_mpe",&flg1);CHKERRQ(ierr);
  if (flg1) PetscLogMPEBegin();
#endif
  flg1 = PETSC_FALSE;
  flg2 = PETSC_FALSE;
  flg3 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-log_all",&flg1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-log",&flg2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_summary",&flg3);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_summary_python",&flg4);CHKERRQ(ierr);
  if (flg1)                      {  ierr = PetscLogAllBegin();CHKERRQ(ierr); }
  else if (flg2 || flg3 || flg4) {  ierr = PetscLogBegin();CHKERRQ(ierr);}
    
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_trace",mname,250,&flg1);CHKERRQ(ierr);
  if (flg1) { 
    char name[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    FILE *file;
    if (mname[0]) {
      sprintf(name,"%s.%d",mname,rank);
      ierr = PetscFixFilename(name,fname);CHKERRQ(ierr);
      file = fopen(fname,"w"); 
      if (!file) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open trace file: %s",fname);
    } else {
      file = PETSC_STDOUT;
    }
    ierr = PetscLogTraceBegin(file);CHKERRQ(ierr);
  }
#endif

  /*
      Setup building of stack frames for all function calls
  */
#if defined(PETSC_USE_DEBUG)
  PetscThreadLocalRegister(petscstack); /* Creates petscstack_key if needed */
  ierr = PetscStackCreate();CHKERRQ(ierr);
#endif

  ierr = PetscOptionsGetBool(PETSC_NULL,"-options_gui",&PetscOptionsPublish,PETSC_NULL);CHKERRQ(ierr);

#if defined(PETSC_HAVE_CUDA)
  ierr = PetscOptionsHasName(PETSC_NULL,"-cuda_show_devices",&flg1);CHKERRQ(ierr);
  if (flg1) {
    struct cudaDeviceProp prop;
    int devCount;
    int device;

    ierr = cudaGetDeviceCount(&devCount);CHKERRQ(ierr);
    for(device = 0; device < devCount; ++device) {
      ierr = cudaGetDeviceProperties(&prop, device);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "CUDA device %d: %s\n", device, prop.name);CHKERRQ(ierr);
    }
  }
  {
    int size;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    if (size>1) {
      int devCount, device, rank;
      ierr = cudaGetDeviceCount(&devCount);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
      device = rank % devCount;
      ierr = cudaSetDevice(device);CHKERRQ(ierr);        
    }
    else {
      int device;
      /* the code below works for serial GPU simulations */
      ierr = PetscOptionsGetInt(PETSC_NULL,"-cuda_set_device", &device, &flg1);CHKERRQ(ierr);
      if (flg1) {
        ierr = cudaSetDevice(device);CHKERRQ(ierr);
      }
    }
  }
#endif


#if defined(PETSC_HAVE_PTHREADCLASSES)
  ierr = PetscOptionsCheckInitial_Private_Pthread();
#endif
  /*
       Print basic help message
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = (*PetscHelpPrintf)(comm,"Options for all PETSc programs:\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -help: prints help method for each option\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -on_error_abort: cause an abort when an error is detected. Useful \n ");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       only when run in the debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -on_error_attach_debugger [gdb,dbx,xxgdb,ups,noxterm]\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       start the debugger in new xterm\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       unless noxterm is given\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -start_in_debugger [gdb,dbx,xxgdb,ups,noxterm]\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"       start all processes in the debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -on_error_emacs <machinename>\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"    emacs jumps to error file\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -debugger_nodes [n1,n2,..] Nodes to start in debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -debugger_pause [m] : delay (in seconds) to attach debugger\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -stop_for_debugger : prints message on how to attach debugger manually\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"                      waits the delay for you to attach\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -display display: Location where graphics and debuggers are displayed\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -no_signal_handler: do not trap error signals\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -mpi_return_on_error: MPI returns error code, rather than abort on internal error\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -fp_trap: stop on floating point exceptions\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"           note on IBM RS6000 this slows run greatly\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_dump <optional filename>: dump list of unfreed memory at conclusion\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc: use our error checking malloc\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc no: don't use error checking malloc\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_info: prints total memory usage\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_log: keeps log of all memory allocations\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -malloc_debug: enables extended checking for memory corruption\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_table: dump list of options inputted\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_left: dump list of unused options\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_left no: don't dump list of unused options\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -tmp tmpdir: alternative /tmp directory\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -shared_tmp: tmp directory is shared by all processors\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -not_shared_tmp: each processor has separate tmp directory\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -memory_info: print memory usage at end of run\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -server <port>: Run PETSc webserver (default port is 8080) see PetscWebServe()\n");CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = (*PetscHelpPrintf)(comm," -get_total_flops: total flops over all processors\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log[_all _summary _summary_python]: logging objects and events\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log_trace [filename]: prints trace of all PETSc calls\n");CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPE)
    ierr = (*PetscHelpPrintf)(comm," -log_mpe: Also create logfile viewable through upshot\n");CHKERRQ(ierr);
#endif
    ierr = (*PetscHelpPrintf)(comm," -info <optional filename>: print informative messages about the calculations\n");CHKERRQ(ierr);
#endif
    ierr = (*PetscHelpPrintf)(comm," -v: prints PETSc version number and release date\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -options_file <file>: reads options from file\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -petsc_sleep n: sleeps n seconds before running program\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"-----------------------------------------------\n");CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetReal(PETSC_NULL,"-petsc_sleep",&si,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscSleep(si);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-info_exclude",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  ierr = PetscStrstr(mname,"null",&f);CHKERRQ(ierr);
  if (f) {
    ierr = PetscInfoDeactivateClass(0);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_CUSP)
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_summary",&flg3);CHKERRQ(ierr);
  if (flg3) flg1 = PETSC_TRUE;
  else flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-cusp_synchronize",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) PetscCUSPSynchronize = PETSC_TRUE;
#endif

  PetscFunctionReturn(0);
}

