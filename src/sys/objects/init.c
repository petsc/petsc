#define PETSC_DLL

/*

   This file defines part of the initialization of PETSc

  This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.
*/

#include "petscsys.h"        /*I  "petscsys.h"   I*/
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
/* ------------------------Nasty global variables -------------------------------*/
/*
     Indicates if PETSc started up MPI, or it was 
   already started before PETSc was initialized.
*/
PetscTruth  PETSC_DLLEXPORT PetscBeganMPI         = PETSC_FALSE;
PetscTruth  PETSC_DLLEXPORT PetscInitializeCalled = PETSC_FALSE;
PetscTruth  PETSC_DLLEXPORT PetscFinalizeCalled   = PETSC_FALSE;
PetscMPIInt PETSC_DLLEXPORT PetscGlobalRank = -1;
PetscMPIInt PETSC_DLLEXPORT PetscGlobalSize = -1;

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_COMPLEX_INSTANTIATE)
template <> class std::complex<double>; /* instantiate complex template class */
#endif
#if !defined(PETSC_HAVE_MPI_C_DOUBLE_COMPLEX)
MPI_Datatype  PETSC_DLLEXPORT MPI_C_DOUBLE_COMPLEX;
MPI_Datatype  PETSC_DLLEXPORT MPI_C_COMPLEX;
#endif
PetscScalar   PETSC_DLLEXPORT PETSC_i; 
#else
PetscScalar   PETSC_DLLEXPORT PETSC_i = 0.0; 
#endif
MPI_Datatype  PETSC_DLLEXPORT MPIU_2SCALAR = 0;
MPI_Datatype  PETSC_DLLEXPORT MPIU_2INT = 0;

#if defined(PETSC_USE_SCALAR_QD_DD)
MPI_Datatype  PETSC_DLLEXPORT MPIU_QD_DD;
#endif
/*
     These are needed by petscbt.h
*/
#include "petscbt.h"
char     PETSC_DLLEXPORT _BT_mask = ' ';
char     PETSC_DLLEXPORT _BT_c = ' ';
PetscInt PETSC_DLLEXPORT _BT_idx  = 0;

/*
       Function that is called to display all error messages
*/
PetscErrorCode PETSC_DLLEXPORT (*PetscErrorPrintf)(const char [],...)          = PetscErrorPrintfDefault;
PetscErrorCode PETSC_DLLEXPORT (*PetscHelpPrintf)(MPI_Comm,const char [],...)  = PetscHelpPrintfDefault;
PetscErrorCode PETSC_DLLEXPORT (*PetscVFPrintf)(FILE*,const char[],va_list)    = PetscVFPrintfDefault;

/* ------------------------------------------------------------------------------*/
/* 
   Optional file where all PETSc output from various prints is saved
*/
FILE *petsc_history = PETSC_NULL;

#undef __FUNCT__  
#define __FUNCT__ "PetscLogOpenHistoryFile"
PetscErrorCode PETSC_DLLEXPORT PetscLogOpenHistoryFile(const char filename[],FILE **fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  char           pfile[PETSC_MAX_PATH_LEN],pname[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN],date[64];
  char           version[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    char arch[10];
    int  err;

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

    *fd = fopen(fname,"a"); if (!fd) SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file: %s",fname);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"---------------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"%s %s\n",version,date);CHKERRQ(ierr);
    ierr = PetscGetProgramName(pname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"%s on a %s, %d proc. with options:\n",pname,arch,size);CHKERRQ(ierr);
    ierr = PetscOptionsPrint(*fd);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,*fd,"---------------------------------------------------------\n");CHKERRQ(ierr);
    err = fflush(*fd);
    if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on file");        
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogCloseHistoryFile"
PetscErrorCode PETSC_DLLEXPORT PetscLogCloseHistoryFile(FILE **fd)
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
    if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on file");        
    err = fclose(*fd);
    if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");        
  }
  PetscFunctionReturn(0); 
}

/* ------------------------------------------------------------------------------*/

/* 
   This is ugly and probably belongs somewhere else, but I want to 
  be able to put a true MPI abort error handler with command line args.

    This is so MPI errors in the debugger will leave all the stack 
  frames. The default abort cleans up and exits.
*/

#undef __FUNCT__  
#define __FUNCT__ "Petsc_MPI_AbortOnError"
void Petsc_MPI_AbortOnError(MPI_Comm *comm,PetscMPIInt *flag) 
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("MPI error %d\n",(int)*flag);
  abort();
}

#undef __FUNCT__  
#define __FUNCT__ "Petsc_MPI_DebuggerOnError"
void Petsc_MPI_DebuggerOnError(MPI_Comm *comm,PetscMPIInt *flag) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  (*PetscErrorPrintf)("MPI error %d\n",(int)*flag);
  ierr = PetscAttachDebugger();
  if (ierr) { /* hopeless so get out */
    MPI_Finalize();
    exit(*flag);
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

.seealso: PetscInitialize(), PetscOptionsPrint(), PetscMallocDump(), PetscMPIDump(), PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscEnd(void)
{
  PetscFunctionBegin;
  PetscFinalize();
  exit(0);
  return 0;
}

PetscTruth   PetscOptionsPublish = PETSC_FALSE;
EXTERN PetscErrorCode        PetscSetUseTrMalloc_Private(void);
extern PetscTruth petscsetmallocvisited;
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
PetscErrorCode PETSC_DLLEXPORT PetscSetHelpVersionFunctions(PetscErrorCode (*help)(MPI_Comm),PetscErrorCode (*version)(MPI_Comm))
{
  PetscFunctionBegin;
  PetscExternalHelpFunction    = help;
  PetscExternalVersionFunction = version;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCheckInitial_Private"
PetscErrorCode PETSC_DLLEXPORT PetscOptionsCheckInitial_Private(void)
{
  char           string[64],mname[PETSC_MAX_PATH_LEN],*f;
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscTruth     flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE,flag,flgz,flgzout;
  PetscErrorCode ierr;
  PetscReal      si;
  int            i;
  PetscMPIInt    rank;
  char           version[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /*
      Setup the memory management; support for tracing malloc() usage 
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-malloc_log",&flg3);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-malloc",&flg1,&flg2);CHKERRQ(ierr);
  if ((!flg2 || flg1) && !petscsetmallocvisited) {
    ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr); 
  }
#else
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-malloc_dump",&flg1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-malloc",&flg2,PETSC_NULL);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3) {ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr);}
#endif
  if (flg3) {
    ierr = PetscMallocSetDumpLog();CHKERRQ(ierr); 
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-malloc_debug",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) { 
    ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr);
    ierr = PetscMallocDebug(PETSC_TRUE);CHKERRQ(ierr);
  }

  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-malloc_info",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (!flg1) {
    flg1 = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-memory_info",&flg1,PETSC_NULL);CHKERRQ(ierr);
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
    ierr = (*PetscHelpPrintf)(comm,"See docs/copyright.html for copyright information\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/changes/index.html for recent updates.\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"See docs/troubleshooting.html for problems.\n");CHKERRQ(ierr);
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
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-fp_trap",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) { ierr = PetscSetFPTrap(PETSC_FP_TRAP_ON);CHKERRQ(ierr); }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-on_error_abort",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) { ierr = PetscPushErrorHandler(PetscAbortErrorHandler,0);CHKERRQ(ierr)} 
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-on_error_mpiabort",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) { ierr = PetscPushErrorHandler(PetscMPIAbortErrorHandler,0);CHKERRQ(ierr)}
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-mpi_return_on_error",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) {
    ierr = MPI_Errhandler_set(comm,MPI_ERRORS_RETURN);CHKERRQ(ierr);
  }
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-no_signal_handler",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (!flg1) { ierr = PetscPushSignalHandler(PetscDefaultSignalHandler,(void*)0);CHKERRQ(ierr) }

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
  if (flg1 && !rank) {ierr = PetscPushErrorHandler(PetscEmacsClientErrorHandler,emacsmachinename);CHKERRQ(ierr)}

#if defined(PETSC_USE_SOCKET_VIEWER)
  /*
    Activates new sockets for zope if needed
  */
  ierr=PetscOptionsHasName(PETSC_NULL,"-zope", &flgz);CHKERRQ(ierr);
  ierr=PetscOptionsHasName(PETSC_NULL,"-nostdout", &flgzout);CHKERRQ(ierr);
  if(flgz){
    extern FILE* PETSC_ZOPEFD;
    int sockfd; 
    char hostname[256];
    char username[256];
    int remoteport = 9999;
    ierr=PetscOptionsGetString(PETSC_NULL, "-zope", hostname, 256, &flgz);CHKERRQ(ierr);
    if(!hostname[0]){
      ierr=PetscGetHostName(hostname,256);CHKERRQ(ierr);}
    ierr=PetscOpenSocket(hostname, remoteport, &sockfd);CHKERRQ(ierr);
    ierr = PetscGetUserName(username, 256);
    PETSC_ZOPEFD = fdopen(sockfd, "w");
    if(flgzout){
      PETSC_STDOUT = PETSC_ZOPEFD;
      fprintf(PETSC_STDOUT, "<<<user>>> %s\n",username);
      fprintf(PETSC_STDOUT, "<<<start>>>");
    }
    else{
      fprintf(PETSC_ZOPEFD, "<<<user>>> %s\n",username);
      fprintf(PETSC_ZOPEFD, "<<<start>>>");
    }}
#endif

  /*
        Setup profiling and logging
  */
#if defined (PETSC_USE_INFO)
  { 
    char logname[PETSC_MAX_PATH_LEN]; logname[0] = 0;
    ierr = PetscOptionsGetString(PETSC_NULL,"-info",logname,250,&flg1);CHKERRQ(ierr);
    if (flg1 && logname[0]) {
      PetscInfoAllow(PETSC_TRUE,logname); 
    } else if (flg1) {
      PetscInfoAllow(PETSC_TRUE,PETSC_NULL); 
    }
  }
#endif
#if defined(PETSC_USE_LOG)
  mname[0] = 0;
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_history",mname,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  if (flg1) {
    if (mname[0]) {
      ierr = PetscLogOpenHistoryFile(mname,&petsc_history);CHKERRQ(ierr);
    } else {
      ierr = PetscLogOpenHistoryFile(0,&petsc_history);CHKERRQ(ierr);
    }
  }
#if defined(PETSC_HAVE_MPE)
  flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-log_mpe",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) PetscLogMPEBegin();
#endif
  flg1 = PETSC_FALSE;
  flg2 = PETSC_FALSE;
  flg3 = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-log_all",&flg1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-log",&flg2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_summary",&flg3);CHKERRQ(ierr);
  if (flg1)              {  ierr = PetscLogAllBegin();CHKERRQ(ierr); }
  else if (flg2 || flg3) {  ierr = PetscLogBegin();CHKERRQ(ierr);}
    
  ierr = PetscOptionsGetString(PETSC_NULL,"-log_trace",mname,250,&flg1);CHKERRQ(ierr);
  if (flg1) { 
    char name[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    FILE *file;
    if (mname[0]) {
      sprintf(name,"%s.%d",mname,rank);
      ierr = PetscFixFilename(name,fname);CHKERRQ(ierr);
      file = fopen(fname,"w"); 
      if (!file) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Unable to open trace file: %s",fname);
      }
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
  ierr = PetscStackCreate();CHKERRQ(ierr);
#endif

  ierr = PetscOptionsGetTruth(PETSC_NULL,"-options_gui",&PetscOptionsPublish,PETSC_NULL);CHKERRQ(ierr);

  /*
       Print basic help message
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-help",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = (*PetscHelpPrintf)(comm,"Options for all PETSc programs:\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -help: prints help method for each option");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -on_error_abort: cause an abort when an error is");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," detected. Useful \n       only when run in the debugger\n");CHKERRQ(ierr);
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
#if defined(PETSC_USE_LOG)
    ierr = (*PetscHelpPrintf)(comm," -get_total_flops: total flops over all processors\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm," -log[_all _summary]: logging objects and events\n");CHKERRQ(ierr);
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
    ierr = PetscInfoDeactivateClass(PETSC_NULL);CHKERRQ(ierr);
  }


  PetscFunctionReturn(0);
}


