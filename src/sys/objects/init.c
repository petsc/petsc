/*

   This file defines part of the initialization of PETSc

  This file uses regular malloc and free because it cannot know 
  what malloc is being used until it has already processed the input.
*/

#include <petscsys.h>        /*I  "petscsys.h"   I*/

#define PETSC_PTHREADCLASSES_DEBUG 0

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#if defined(PETSC_HAVE_SCHED_H)
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif
#if defined(PETSC_HAVE_PTHREAD_H)
#include <pthread.h>
#endif

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
#include <valgrind/valgrind.h>
#endif

/* ------------------------Nasty global variables -------------------------------*/
/*
     Indicates if PETSc started up MPI, or it was 
   already started before PETSc was initialized.
*/
PetscBool    PetscBeganMPI         = PETSC_FALSE;
PetscBool    PetscInitializeCalled = PETSC_FALSE;
PetscBool    PetscFinalizeCalled   = PETSC_FALSE;
PetscBool    PetscUseThreadPool    = PETSC_FALSE;
PetscBool    PetscThreadGo         = PETSC_TRUE;
PetscMPIInt  PetscGlobalRank = -1;
PetscMPIInt  PetscGlobalSize = -1;

#if defined(PETSC_HAVE_PTHREADCLASSES)
PetscMPIInt  PetscMaxThreads = 2;
pthread_t*   PetscThreadPoint;
#define PETSC_HAVE_PTHREAD_BARRIER
#if defined(PETSC_HAVE_PTHREAD_BARRIER)
pthread_barrier_t* BarrPoint;   /* used by 'true' thread pool */
#endif
PetscErrorCode ithreaderr = 0;
int*         pVal;

#define CACHE_LINE_SIZE 64  /* used by 'chain', 'main','tree' thread pools */
int* ThreadCoreAffinity;

typedef enum {JobInitiated,ThreadsWorking,JobCompleted} estat;  /* used by 'chain','tree' thread pool */

typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  void* (*pfunc)(void*);
  void** pdata;
  PetscBool startJob;
  estat eJobStat;
  PetscBool** arrThreadStarted;
  PetscBool** arrThreadReady;
} sjob_tree;
sjob_tree job_tree;
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  void* (*pfunc)(void*);
  void** pdata;
  PetscBool** arrThreadReady;
} sjob_main;
sjob_main job_main;
typedef struct {
  pthread_mutex_t** mutexarray;
  pthread_cond_t**  cond1array;
  pthread_cond_t** cond2array;
  void* (*pfunc)(void*);
  void** pdata;
  PetscBool startJob;
  estat eJobStat;
  PetscBool** arrThreadStarted;
  PetscBool** arrThreadReady;
} sjob_chain;
sjob_chain job_chain;
#if defined(PETSC_HAVE_PTHREAD_BARRIER)
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  void* (*pfunc)(void*);
  void** pdata;
  pthread_barrier_t* pbarr;
  int iNumJobThreads;
  int iNumReadyThreads;
  PetscBool startJob;
} sjob_true;
sjob_true job_true = {PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER,NULL,NULL,NULL,0,0,PETSC_FALSE};
#endif

pthread_cond_t  main_cond  = PTHREAD_COND_INITIALIZER;  /* used by 'true', 'chain','tree' thread pools */
char* arrmutex; /* used by 'chain','main','tree' thread pools */
char* arrcond1; /* used by 'chain','main','tree' thread pools */
char* arrcond2; /* used by 'chain','main','tree' thread pools */
char* arrstart; /* used by 'chain','main','tree' thread pools */
char* arrready; /* used by 'chain','main','tree' thread pools */

/* Function Pointers */
void*          (*PetscThreadFunc)(void*) = NULL;
void*          (*PetscThreadInitialize)(PetscInt) = NULL;
PetscErrorCode (*PetscThreadFinalize)(void) = NULL;
void           (*MainWait)(void) = NULL;
PetscErrorCode (*MainJob)(void* (*pFunc)(void*),void**,PetscInt) = NULL;
/**** Tree Thread Pool Functions ****/
void*          PetscThreadFunc_Tree(void*);
void*          PetscThreadInitialize_Tree(PetscInt);
PetscErrorCode PetscThreadFinalize_Tree(void);
void           MainWait_Tree(void);
PetscErrorCode MainJob_Tree(void* (*pFunc)(void*),void**,PetscInt);
/**** Main Thread Pool Functions ****/
void*          PetscThreadFunc_Main(void*);
void*          PetscThreadInitialize_Main(PetscInt);
PetscErrorCode PetscThreadFinalize_Main(void);
void           MainWait_Main(void);
PetscErrorCode MainJob_Main(void* (*pFunc)(void*),void**,PetscInt);
/**** Chain Thread Pool Functions ****/
void*          PetscThreadFunc_Chain(void*);
void*          PetscThreadInitialize_Chain(PetscInt);
PetscErrorCode PetscThreadFinalize_Chain(void);
void           MainWait_Chain(void);
PetscErrorCode MainJob_Chain(void* (*pFunc)(void*),void**,PetscInt);
/**** True Thread Pool Functions ****/
void*          PetscThreadFunc_True(void*);
void*          PetscThreadInitialize_True(PetscInt);
PetscErrorCode PetscThreadFinalize_True(void);
void           MainWait_True(void);
PetscErrorCode MainJob_True(void* (*pFunc)(void*),void**,PetscInt);
/**** NO Thread Pool Function  ****/
PetscErrorCode MainJob_Spawn(void* (*pFunc)(void*),void**,PetscInt);
/****  ****/
void* FuncFinish(void*);
void* PetscThreadRun(MPI_Comm Comm,void* (*pFunc)(void*),int,pthread_t*,void**);
void* PetscThreadStop(MPI_Comm Comm,int,pthread_t*);
#endif

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
     These are needed by petscbt.h
*/
#include <petscbt.h>
char      _BT_mask = ' ';
char      _BT_c = ' ';
PetscInt  _BT_idx  = 0;

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
  This is needed to turn on/off cusp synchronization */
PetscBool   synchronizeCUSP = PETSC_FALSE;

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

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsCheckInitial_Private"
PetscErrorCode  PetscOptionsCheckInitial_Private(void)
{
  char           string[64],mname[PETSC_MAX_PATH_LEN],*f;
  MPI_Comm       comm = PETSC_COMM_WORLD;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE,flg4 = PETSC_FALSE,flag,flgz,flgzout;
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
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_USE_PTHREAD)
  ierr = PetscOptionsGetBool(PETSC_NULL,"-malloc",&flg1,&flg2);CHKERRQ(ierr);
  if ((!flg2 || flg1) && !petscsetmallocvisited) {
#if defined(PETSC_HAVE_VALGRIND)
    if (flg2 || !(RUNNING_ON_VALGRIND)) {
      /* turn off default -malloc if valgrind is being used */
#endif
      ierr = PetscSetUseTrMalloc_Private();CHKERRQ(ierr); 
#if defined(PETSC_HAVE_VALGRIND)
    }
#endif
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
  if (flg1) { ierr = PetscPushErrorHandler(PetscAbortErrorHandler,0);CHKERRQ(ierr);}
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

#if defined(PETSC_USE_SOCKET_VIEWER)
  /*
    Activates new sockets for zope if needed
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-zope", &flgz);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-nostdout", &flgzout);CHKERRQ(ierr);
  if (flgz){
    int  sockfd;
    char hostname[256];
    char username[256];
    int  remoteport = 9999;

    ierr = PetscOptionsGetString(PETSC_NULL, "-zope", hostname, 256, &flgz);CHKERRQ(ierr);
    if (!hostname[0]){
      ierr = PetscGetHostName(hostname,256);CHKERRQ(ierr);
    }
    ierr = PetscOpenSocket(hostname, remoteport, &sockfd);CHKERRQ(ierr);
    ierr = PetscGetUserName(username, 256);CHKERRQ(ierr);
    PETSC_ZOPEFD = fdopen(sockfd, "w");
    if (flgzout){
      PETSC_STDOUT = PETSC_ZOPEFD;
      fprintf(PETSC_STDOUT, "<<<user>>> %s\n",username);
      fprintf(PETSC_STDOUT, "<<<start>>>");
    } else {
      fprintf(PETSC_ZOPEFD, "<<<user>>> %s\n",username);
      fprintf(PETSC_ZOPEFD, "<<<start>>>");
    }
  }
#endif
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
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_USE_PTHREAD)
  ierr = PetscStackCreate();CHKERRQ(ierr);
#endif

  ierr = PetscOptionsGetBool(PETSC_NULL,"-options_gui",&PetscOptionsPublish,PETSC_NULL);CHKERRQ(ierr);

#if defined(PETSC_HAVE_PTHREADCLASSES)
  /*
      Determine whether user specified maximum number of threads
   */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-thread_max",&PetscMaxThreads,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-main",&flg1);CHKERRQ(ierr);
  if(flg1) {
    cpu_set_t mset;
    int icorr,ncorr = get_nprocs();
    ierr = PetscOptionsGetInt(PETSC_NULL,"-main",&icorr,PETSC_NULL);CHKERRQ(ierr);
    CPU_ZERO(&mset);
    CPU_SET(icorr%ncorr,&mset);
    sched_setaffinity(0,sizeof(cpu_set_t),&mset);
  }

  PetscInt N_CORES = get_nprocs();
  ThreadCoreAffinity = (int*)malloc(N_CORES*sizeof(int));
  char tstr[9];
  char tbuf[2];
  strcpy(tstr,"-thread");
  for(i=0;i<PetscMaxThreads;i++) {
    ThreadCoreAffinity[i] = i;
    sprintf(tbuf,"%d",i);
    strcat(tstr,tbuf);
    ierr = PetscOptionsHasName(PETSC_NULL,tstr,&flg1);CHKERRQ(ierr);
    if(flg1) {
      ierr = PetscOptionsGetInt(PETSC_NULL,tstr,&ThreadCoreAffinity[i],PETSC_NULL);CHKERRQ(ierr);
      ThreadCoreAffinity[i] = ThreadCoreAffinity[i]%N_CORES; /* check on the user */
    }
    tstr[7] = '\0';
  }

  /*
      Determine whether to use thread pool
   */
  ierr = PetscOptionsHasName(PETSC_NULL,"-use_thread_pool",&flg1);CHKERRQ(ierr);
  if (flg1) {
    PetscUseThreadPool = PETSC_TRUE;
    /* get the thread pool type */
    PetscInt ipool = 0;
    const char *choices[4] = {"true","tree","main","chain"};

    ierr = PetscOptionsGetEList(PETSC_NULL,"-use_thread_pool",choices,4,&ipool,PETSC_NULL);CHKERRQ(ierr);
    switch(ipool) {
    case 1:
      PetscThreadFunc       = &PetscThreadFunc_Tree;
      PetscThreadInitialize = &PetscThreadInitialize_Tree;
      PetscThreadFinalize   = &PetscThreadFinalize_Tree;
      MainWait              = &MainWait_Tree;
      MainJob               = &MainJob_Tree;
      PetscInfo(PETSC_NULL,"Using tree thread pool\n");
      break;
    case 2:
      PetscThreadFunc       = &PetscThreadFunc_Main;
      PetscThreadInitialize = &PetscThreadInitialize_Main;
      PetscThreadFinalize   = &PetscThreadFinalize_Main;
      MainWait              = &MainWait_Main;
      MainJob               = &MainJob_Main;
      PetscInfo(PETSC_NULL,"Using main thread pool\n");
      break;
#if defined(PETSC_HAVE_PTHREAD_BARRIER)
    case 3:
#else
    default:
#endif
      PetscThreadFunc       = &PetscThreadFunc_Chain;
      PetscThreadInitialize = &PetscThreadInitialize_Chain;
      PetscThreadFinalize   = &PetscThreadFinalize_Chain;
      MainWait              = &MainWait_Chain;
      MainJob               = &MainJob_Chain;
      PetscInfo(PETSC_NULL,"Using chain thread pool\n");
      break;
#if defined(PETSC_HAVE_PTHREAD_BARRIER)
    default:
      PetscThreadFunc       = &PetscThreadFunc_True;
      PetscThreadInitialize = &PetscThreadInitialize_True;
      PetscThreadFinalize   = &PetscThreadFinalize_True;
      MainWait              = &MainWait_True;
      MainJob               = &MainJob_True;
      PetscInfo(PETSC_NULL,"Using true thread pool\n");
      break;
#endif
    }
    PetscThreadInitialize(PetscMaxThreads);
  } else {
    /* need to define these in the case on 'no threads' or 'thread create/destroy' */
    /* could take any of the above versions */
    MainJob               = &MainJob_Spawn;
  }
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
    ierr = PetscInfoDeactivateClass(PETSC_NULL);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_CUSP)
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_summary",&flg3);CHKERRQ(ierr);
  if (flg3) flg1 = PETSC_TRUE;
  else flg1 = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-cusp_synchronize",&flg1,PETSC_NULL);CHKERRQ(ierr);
  if (flg1) synchronizeCUSP = PETSC_TRUE;
#endif

  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PTHREADCLASSES)

/**** 'Tree' Thread Pool Functions ****/
void* PetscThreadFunc_Tree(void* arg) {
  PetscErrorCode iterr;
  int icorr,ierr;
  int* pId = (int*)arg;
  int ThreadId = *pId,Mary = 2,i,SubWorker;
  PetscBool PeeOn;
  cpu_set_t mset;
  if (PETSC_PTHREADCLASSES_DEBUG) printf("Thread %d In Tree Thread Function\n",ThreadId);
  icorr = ThreadCoreAffinity[ThreadId];
  CPU_ZERO(&mset);
  CPU_SET(icorr,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);

  if((Mary*ThreadId+1)>(PetscMaxThreads-1)) {
    PeeOn = PETSC_TRUE;
  }
  else {
    PeeOn = PETSC_FALSE;
  }
  if(PeeOn==PETSC_FALSE) {
    /* check your subordinates, wait for them to be ready */
    for(i=1;i<=Mary;i++) {
      SubWorker = Mary*ThreadId+i;
      if(SubWorker<PetscMaxThreads) {
        ierr = pthread_mutex_lock(job_tree.mutexarray[SubWorker]);
        while(*(job_tree.arrThreadReady[SubWorker])==PETSC_FALSE) {
          /* upon entry, automically releases the lock and blocks
           upon return, has the lock */
          ierr = pthread_cond_wait(job_tree.cond1array[SubWorker],job_tree.mutexarray[SubWorker]);
        }
        ierr = pthread_mutex_unlock(job_tree.mutexarray[SubWorker]);
      }
    }
    /* your subordinates are now ready */
  }
  ierr = pthread_mutex_lock(job_tree.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_tree.arrThreadReady[ThreadId]) = PETSC_TRUE;
  if(ThreadId==0) {
    job_tree.eJobStat = JobCompleted;
    /* ignal main */
    ierr = pthread_cond_signal(&main_cond);
  }
  else {
    /* tell your boss that you're ready to work */
    ierr = pthread_cond_signal(job_tree.cond1array[ThreadId]);
  }
  /* the while loop needs to have an exit
  the 'main' thread can terminate all the threads by performing a broadcast
   and calling FuncFinish */
  while(PetscThreadGo) {
    /*need to check the condition to ensure we don't have to wait
      waiting when you don't have to causes problems
     also need to check the condition to ensure proper handling of spurious wakeups */
    while(*(job_tree.arrThreadReady[ThreadId])==PETSC_TRUE) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
        ierr = pthread_cond_wait(job_tree.cond2array[ThreadId],job_tree.mutexarray[ThreadId]);
	*(job_tree.arrThreadStarted[ThreadId]) = PETSC_TRUE;
	*(job_tree.arrThreadReady[ThreadId])   = PETSC_FALSE;
    }
    if(ThreadId==0) {
      job_tree.startJob = PETSC_FALSE;
      job_tree.eJobStat = ThreadsWorking;
    }
    ierr = pthread_mutex_unlock(job_tree.mutexarray[ThreadId]);
    if(PeeOn==PETSC_FALSE) {
      /* tell your subordinates it's time to get to work */
      for(i=1; i<=Mary; i++) {
	SubWorker = Mary*ThreadId+i;
        if(SubWorker<PetscMaxThreads) {
          ierr = pthread_cond_signal(job_tree.cond2array[SubWorker]);
        }
      }
    }
    /* do your job */
    if(job_tree.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_tree.pfunc(job_tree.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_tree.pfunc(job_tree.pdata[ThreadId]);
    }
    if(iterr!=0) {
      ithreaderr = 1;
    }
    if(PetscThreadGo) {
      /* reset job, get ready for more */
      if(PeeOn==PETSC_FALSE) {
        /* check your subordinates, waiting for them to be ready
         how do you know for a fact that a given subordinate has actually started? */
	for(i=1;i<=Mary;i++) {
	  SubWorker = Mary*ThreadId+i;
          if(SubWorker<PetscMaxThreads) {
            ierr = pthread_mutex_lock(job_tree.mutexarray[SubWorker]);
            while(*(job_tree.arrThreadReady[SubWorker])==PETSC_FALSE||*(job_tree.arrThreadStarted[SubWorker])==PETSC_FALSE) {
              /* upon entry, automically releases the lock and blocks
               upon return, has the lock */
              ierr = pthread_cond_wait(job_tree.cond1array[SubWorker],job_tree.mutexarray[SubWorker]);
            }
            ierr = pthread_mutex_unlock(job_tree.mutexarray[SubWorker]);
          }
	}
        /* your subordinates are now ready */
      }
      ierr = pthread_mutex_lock(job_tree.mutexarray[ThreadId]);
      *(job_tree.arrThreadReady[ThreadId]) = PETSC_TRUE;
      if(ThreadId==0) {
	job_tree.eJobStat = JobCompleted; /* oot thread: last thread to complete, guaranteed! */
        /* root thread signals 'main' */
        ierr = pthread_cond_signal(&main_cond);
      }
      else {
        /* signal your boss before you go to sleep */
        ierr = pthread_cond_signal(job_tree.cond1array[ThreadId]);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_Tree"
void* PetscThreadInitialize_Tree(PetscInt N) {
  PetscInt i,ierr;
  int status;

  if(PetscUseThreadPool) {
    size_t Val1 = (size_t)CACHE_LINE_SIZE;
    size_t Val2 = (size_t)PetscMaxThreads*CACHE_LINE_SIZE;
    arrmutex = (char*)memalign(Val1,Val2);
    arrcond1 = (char*)memalign(Val1,Val2);
    arrcond2 = (char*)memalign(Val1,Val2);
    arrstart = (char*)memalign(Val1,Val2);
    arrready = (char*)memalign(Val1,Val2);
    job_tree.mutexarray       = (pthread_mutex_t**)malloc(PetscMaxThreads*sizeof(pthread_mutex_t*));
    job_tree.cond1array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
    job_tree.cond2array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
    job_tree.arrThreadStarted = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
    job_tree.arrThreadReady   = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
    /* initialize job structure */
    for(i=0; i<PetscMaxThreads; i++) {
      job_tree.mutexarray[i]        = (pthread_mutex_t*)(arrmutex+CACHE_LINE_SIZE*i);
      job_tree.cond1array[i]        = (pthread_cond_t*)(arrcond1+CACHE_LINE_SIZE*i);
      job_tree.cond2array[i]        = (pthread_cond_t*)(arrcond2+CACHE_LINE_SIZE*i);
      job_tree.arrThreadStarted[i]  = (PetscBool*)(arrstart+CACHE_LINE_SIZE*i);
      job_tree.arrThreadReady[i]    = (PetscBool*)(arrready+CACHE_LINE_SIZE*i);
    }
    for(i=0; i<PetscMaxThreads; i++) {
      ierr = pthread_mutex_init(job_tree.mutexarray[i],NULL);
      ierr = pthread_cond_init(job_tree.cond1array[i],NULL);
      ierr = pthread_cond_init(job_tree.cond2array[i],NULL);
      *(job_tree.arrThreadStarted[i])  = PETSC_FALSE;
      *(job_tree.arrThreadReady[i])    = PETSC_FALSE;
    }
    job_tree.pfunc = NULL;
    job_tree.pdata = (void**)malloc(N*sizeof(void*));
    job_tree.startJob = PETSC_FALSE;
    job_tree.eJobStat = JobInitiated;
    pVal = (int*)malloc(N*sizeof(int));
    /* allocate memory in the heap for the thread structure */
    PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
    /* create threads */
    for(i=0; i<N; i++) {
      pVal[i] = i;
      status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
      /* should check status */
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_Tree"
PetscErrorCode PetscThreadFinalize_Tree() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
    /* do error checking*/
  }
  free(PetscThreadPoint);
  free(arrmutex);
  free(arrcond1);
  free(arrcond2);
  free(arrstart);
  free(arrready);
  free(job_tree.pdata);
  free(pVal);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Tree"
void MainWait_Tree() {
  int ierr;
  ierr = pthread_mutex_lock(job_tree.mutexarray[0]);
  while(job_tree.eJobStat<JobCompleted||job_tree.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond,job_tree.mutexarray[0]);
  }
  ierr = pthread_mutex_unlock(job_tree.mutexarray[0]);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_Tree"
PetscErrorCode MainJob_Tree(void* (*pFunc)(void*),void** data,PetscInt n) {
  int i,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait();
  job_tree.pfunc = pFunc;
  job_tree.pdata = data;
  job_tree.startJob = PETSC_TRUE;
  for(i=0; i<PetscMaxThreads; i++) {
    *(job_tree.arrThreadStarted[i]) = PETSC_FALSE;
  }
  job_tree.eJobStat = JobInitiated;
  ierr = pthread_cond_signal(job_tree.cond2array[0]);
  if(pFunc!=FuncFinish) {
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}
/****  ****/

/**** 'Main' Thread Pool Functions ****/
void* PetscThreadFunc_Main(void* arg) {
  PetscErrorCode iterr;
  int icorr,ierr;
  int* pId = (int*)arg;
  int ThreadId = *pId;
  cpu_set_t mset;
  if (PETSC_PTHREADCLASSES_DEBUG) printf("Thread %d In Main Thread Function\n",ThreadId);
  icorr = ThreadCoreAffinity[ThreadId];
  CPU_ZERO(&mset);
  CPU_SET(icorr,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);

  ierr = pthread_mutex_lock(job_main.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_main.arrThreadReady[ThreadId]) = PETSC_TRUE;
  /* tell the BOSS that you're ready to work before you go to sleep */
  ierr = pthread_cond_signal(job_main.cond1array[ThreadId]);

  /* the while loop needs to have an exit
     the 'main' thread can terminate all the threads by performing a broadcast
     and calling FuncFinish */
  while(PetscThreadGo) {
    /* need to check the condition to ensure we don't have to wait
       waiting when you don't have to causes problems
     also need to check the condition to ensure proper handling of spurious wakeups */
    while(*(job_main.arrThreadReady[ThreadId])==PETSC_TRUE) {
      /* upon entry, atomically releases the lock and blocks
       upon return, has the lock */
        ierr = pthread_cond_wait(job_main.cond2array[ThreadId],job_main.mutexarray[ThreadId]);
	/* (job_main.arrThreadReady[ThreadId])   = PETSC_FALSE; */
    }
    ierr = pthread_mutex_unlock(job_main.mutexarray[ThreadId]);
    if(job_main.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_main.pfunc(job_main.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_main.pfunc(job_main.pdata[ThreadId]);
    }
    if(iterr!=0) {
      ithreaderr = 1;
    }
    if(PetscThreadGo) {
      /* reset job, get ready for more */
      ierr = pthread_mutex_lock(job_main.mutexarray[ThreadId]);
      *(job_main.arrThreadReady[ThreadId]) = PETSC_TRUE;
      /* tell the BOSS that you're ready to work before you go to sleep */
      ierr = pthread_cond_signal(job_main.cond1array[ThreadId]);
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_Main"
void* PetscThreadInitialize_Main(PetscInt N) {
  PetscInt i,ierr;
  int status;

  if(PetscUseThreadPool) {
    size_t Val1 = (size_t)CACHE_LINE_SIZE;
    size_t Val2 = (size_t)PetscMaxThreads*CACHE_LINE_SIZE;
    arrmutex = (char*)memalign(Val1,Val2);
    arrcond1 = (char*)memalign(Val1,Val2);
    arrcond2 = (char*)memalign(Val1,Val2);
    arrstart = (char*)memalign(Val1,Val2);
    arrready = (char*)memalign(Val1,Val2);
    job_main.mutexarray       = (pthread_mutex_t**)malloc(PetscMaxThreads*sizeof(pthread_mutex_t*));
    job_main.cond1array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
    job_main.cond2array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
    job_main.arrThreadReady   = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
    /* initialize job structure */
    for(i=0; i<PetscMaxThreads; i++) {
      job_main.mutexarray[i]        = (pthread_mutex_t*)(arrmutex+CACHE_LINE_SIZE*i);
      job_main.cond1array[i]        = (pthread_cond_t*)(arrcond1+CACHE_LINE_SIZE*i);
      job_main.cond2array[i]        = (pthread_cond_t*)(arrcond2+CACHE_LINE_SIZE*i);
      job_main.arrThreadReady[i]    = (PetscBool*)(arrready+CACHE_LINE_SIZE*i);
    }
    for(i=0; i<PetscMaxThreads; i++) {
      ierr = pthread_mutex_init(job_main.mutexarray[i],NULL);
      ierr = pthread_cond_init(job_main.cond1array[i],NULL);
      ierr = pthread_cond_init(job_main.cond2array[i],NULL);
      *(job_main.arrThreadReady[i])    = PETSC_FALSE;
    }
    job_main.pfunc = NULL;
    job_main.pdata = (void**)malloc(N*sizeof(void*));
    pVal = (int*)malloc(N*sizeof(int));
    /* allocate memory in the heap for the thread structure */
    PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
    /* create threads */
    for(i=0; i<N; i++) {
      pVal[i] = i;
      status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
      /* error check */
    }
  }
  else {
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_Main"
PetscErrorCode PetscThreadFinalize_Main() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);CHKERRQ(ierr);
  }
  free(PetscThreadPoint);
  free(arrmutex);
  free(arrcond1);
  free(arrcond2);
  free(arrstart);
  free(arrready);
  free(job_main.pdata);
  free(pVal);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Main"
void MainWait_Main() {
  int i,ierr;
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_mutex_lock(job_main.mutexarray[i]);
    while(*(job_main.arrThreadReady[i])==PETSC_FALSE) {
      ierr = pthread_cond_wait(job_main.cond1array[i],job_main.mutexarray[i]);
    }
    ierr = pthread_mutex_unlock(job_main.mutexarray[i]);
  }
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_Main"
PetscErrorCode MainJob_Main(void* (*pFunc)(void*),void** data,PetscInt n) {
  int i,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait(); /* you know everyone is waiting to be signalled! */
  job_main.pfunc = pFunc;
  job_main.pdata = data;
  for(i=0; i<PetscMaxThreads; i++) {
    *(job_main.arrThreadReady[i]) = PETSC_FALSE; /* why do this?  suppose you get into MainWait first */
  }
  /* tell the threads to go to work */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_cond_signal(job_main.cond2array[i]);
  }
  if(pFunc!=FuncFinish) {
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}
/****  ****/

/**** Chain Thread Functions ****/
void* PetscThreadFunc_Chain(void* arg) {
  PetscErrorCode iterr;
  int icorr,ierr;
  int* pId = (int*)arg;
  int ThreadId = *pId;
  int SubWorker = ThreadId + 1;
  PetscBool PeeOn;
  cpu_set_t mset;
  if (PETSC_PTHREADCLASSES_DEBUG) printf("Thread %d In Chain Thread Function\n",ThreadId);
  icorr = ThreadCoreAffinity[ThreadId];
  CPU_ZERO(&mset);
  CPU_SET(icorr,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);

  if(ThreadId==(PetscMaxThreads-1)) {
    PeeOn = PETSC_TRUE;
  }
  else {
    PeeOn = PETSC_FALSE;
  }
  if(PeeOn==PETSC_FALSE) {
    /* check your subordinate, wait for him to be ready */
    ierr = pthread_mutex_lock(job_chain.mutexarray[SubWorker]);
    while(*(job_chain.arrThreadReady[SubWorker])==PETSC_FALSE) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
      ierr = pthread_cond_wait(job_chain.cond1array[SubWorker],job_chain.mutexarray[SubWorker]);
    }
    ierr = pthread_mutex_unlock(job_chain.mutexarray[SubWorker]);
    /* your subordinate is now ready*/
  }
  ierr = pthread_mutex_lock(job_chain.mutexarray[ThreadId]);
  /* update your ready status */
  *(job_chain.arrThreadReady[ThreadId]) = PETSC_TRUE;
  if(ThreadId==0) {
    job_chain.eJobStat = JobCompleted;
    /* signal main */
    ierr = pthread_cond_signal(&main_cond);
  }
  else {
    /* tell your boss that you're ready to work */
    ierr = pthread_cond_signal(job_chain.cond1array[ThreadId]);
  }
  /*  the while loop needs to have an exit
     the 'main' thread can terminate all the threads by performing a broadcast
   and calling FuncFinish */
  while(PetscThreadGo) {
    /* need to check the condition to ensure we don't have to wait
       waiting when you don't have to causes problems
     also need to check the condition to ensure proper handling of spurious wakeups */
    while(*(job_chain.arrThreadReady[ThreadId])==PETSC_TRUE) {
      /*upon entry, automically releases the lock and blocks
       upon return, has the lock */
        ierr = pthread_cond_wait(job_chain.cond2array[ThreadId],job_chain.mutexarray[ThreadId]);
	*(job_chain.arrThreadStarted[ThreadId]) = PETSC_TRUE;
	*(job_chain.arrThreadReady[ThreadId])   = PETSC_FALSE;
    }
    if(ThreadId==0) {
      job_chain.startJob = PETSC_FALSE;
      job_chain.eJobStat = ThreadsWorking;
    }
    ierr = pthread_mutex_unlock(job_chain.mutexarray[ThreadId]);
    if(PeeOn==PETSC_FALSE) {
      /* tell your subworker it's time to get to work */
      ierr = pthread_cond_signal(job_chain.cond2array[SubWorker]);
    }
    /* do your job */
    if(job_chain.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_chain.pfunc(job_chain.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_chain.pfunc(job_chain.pdata[ThreadId]);
    }
    if(iterr!=0) {
      ithreaderr = 1;
    }
    if(PetscThreadGo) {
      /* reset job, get ready for more */
      if(PeeOn==PETSC_FALSE) {
        /* check your subordinate, wait for him to be ready
         how do you know for a fact that your subordinate has actually started? */
        ierr = pthread_mutex_lock(job_chain.mutexarray[SubWorker]);
        while(*(job_chain.arrThreadReady[SubWorker])==PETSC_FALSE||*(job_chain.arrThreadStarted[SubWorker])==PETSC_FALSE) {
          /* upon entry, automically releases the lock and blocks
           upon return, has the lock */
          ierr = pthread_cond_wait(job_chain.cond1array[SubWorker],job_chain.mutexarray[SubWorker]);
        }
        ierr = pthread_mutex_unlock(job_chain.mutexarray[SubWorker]);
        /* your subordinate is now ready */
      }
      ierr = pthread_mutex_lock(job_chain.mutexarray[ThreadId]);
      *(job_chain.arrThreadReady[ThreadId]) = PETSC_TRUE;
      if(ThreadId==0) {
	job_chain.eJobStat = JobCompleted; /* foreman: last thread to complete, guaranteed! */
        /* root thread (foreman) signals 'main' */
        ierr = pthread_cond_signal(&main_cond);
      }
      else {
        /* signal your boss before you go to sleep */
        ierr = pthread_cond_signal(job_chain.cond1array[ThreadId]);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_Chain"
void* PetscThreadInitialize_Chain(PetscInt N) {
  PetscInt i,ierr;
  int status;

  if(PetscUseThreadPool) {
    size_t Val1 = (size_t)CACHE_LINE_SIZE;
    size_t Val2 = (size_t)PetscMaxThreads*CACHE_LINE_SIZE;
    arrmutex = (char*)memalign(Val1,Val2);
    arrcond1 = (char*)memalign(Val1,Val2);
    arrcond2 = (char*)memalign(Val1,Val2);
    arrstart = (char*)memalign(Val1,Val2);
    arrready = (char*)memalign(Val1,Val2);
    job_chain.mutexarray       = (pthread_mutex_t**)malloc(PetscMaxThreads*sizeof(pthread_mutex_t*));
    job_chain.cond1array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
    job_chain.cond2array       = (pthread_cond_t**)malloc(PetscMaxThreads*sizeof(pthread_cond_t*));
    job_chain.arrThreadStarted = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
    job_chain.arrThreadReady   = (PetscBool**)malloc(PetscMaxThreads*sizeof(PetscBool*));
    /* initialize job structure */
    for(i=0; i<PetscMaxThreads; i++) {
      job_chain.mutexarray[i]        = (pthread_mutex_t*)(arrmutex+CACHE_LINE_SIZE*i);
      job_chain.cond1array[i]        = (pthread_cond_t*)(arrcond1+CACHE_LINE_SIZE*i);
      job_chain.cond2array[i]        = (pthread_cond_t*)(arrcond2+CACHE_LINE_SIZE*i);
      job_chain.arrThreadStarted[i]  = (PetscBool*)(arrstart+CACHE_LINE_SIZE*i);
      job_chain.arrThreadReady[i]    = (PetscBool*)(arrready+CACHE_LINE_SIZE*i);
    }
    for(i=0; i<PetscMaxThreads; i++) {
      ierr = pthread_mutex_init(job_chain.mutexarray[i],NULL);
      ierr = pthread_cond_init(job_chain.cond1array[i],NULL);
      ierr = pthread_cond_init(job_chain.cond2array[i],NULL);
      *(job_chain.arrThreadStarted[i])  = PETSC_FALSE;
      *(job_chain.arrThreadReady[i])    = PETSC_FALSE;
    }
    job_chain.pfunc = NULL;
    job_chain.pdata = (void**)malloc(N*sizeof(void*));
    job_chain.startJob = PETSC_FALSE;
    job_chain.eJobStat = JobInitiated;
    pVal = (int*)malloc(N*sizeof(int));
    /* allocate memory in the heap for the thread structure */
    PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
    /* create threads */
    for(i=0; i<N; i++) {
      pVal[i] = i;
      status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
      /* should check error */
    }
  }
  else {
  }
  return NULL;
}


#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_Chain"
PetscErrorCode PetscThreadFinalize_Chain() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
    /* should check error */
  }
  free(PetscThreadPoint);
  free(arrmutex);
  free(arrcond1);
  free(arrcond2);
  free(arrstart);
  free(arrready);
  free(job_chain.pdata);
  free(pVal);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_Chain"
void MainWait_Chain() {
  int ierr;
  ierr = pthread_mutex_lock(job_chain.mutexarray[0]);
  while(job_chain.eJobStat<JobCompleted||job_chain.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond,job_chain.mutexarray[0]);
  }
  ierr = pthread_mutex_unlock(job_chain.mutexarray[0]);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_Chain"
PetscErrorCode MainJob_Chain(void* (*pFunc)(void*),void** data,PetscInt n) {
  int i,ierr;
  PetscErrorCode ijoberr = 0;

  MainWait();
  job_chain.pfunc = pFunc;
  job_chain.pdata = data;
  job_chain.startJob = PETSC_TRUE;
  for(i=0; i<PetscMaxThreads; i++) {
    *(job_chain.arrThreadStarted[i]) = PETSC_FALSE;
  }
  job_chain.eJobStat = JobInitiated;
  ierr = pthread_cond_signal(job_chain.cond2array[0]);
  if(pFunc!=FuncFinish) {
    MainWait(); /* why wait after? guarantees that job gets done before proceeding with result collection (if any) */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}
/****  ****/

#if defined(PETSC_HAVE_PTHREAD_BARRIER)
/**** True Thread Functions ****/
void* PetscThreadFunc_True(void* arg) {
  int icorr,ierr,iVal;
  int* pId = (int*)arg;
  int ThreadId = *pId;
  PetscErrorCode iterr;
  cpu_set_t mset;
  if (PETSC_PTHREADCLASSES_DEBUG) printf("Thread %d In True Pool Thread Function\n",ThreadId);
  icorr = ThreadCoreAffinity[ThreadId];
  CPU_ZERO(&mset);
  CPU_SET(icorr,&mset);
  sched_setaffinity(0,sizeof(cpu_set_t),&mset);

  ierr = pthread_mutex_lock(&job_true.mutex);
  job_true.iNumReadyThreads++;
  if(job_true.iNumReadyThreads==PetscMaxThreads) {
    ierr = pthread_cond_signal(&main_cond);
  }
  /*the while loop needs to have an exit
    the 'main' thread can terminate all the threads by performing a broadcast
   and calling FuncFinish */
  while(PetscThreadGo) {
    /*need to check the condition to ensure we don't have to wait
      waiting when you don't have to causes problems
     also need to wait if another thread sneaks in and messes with the predicate */
    while(job_true.startJob==PETSC_FALSE&&job_true.iNumJobThreads==0) {
      /* upon entry, automically releases the lock and blocks
       upon return, has the lock */
      if (PETSC_PTHREADCLASSES_DEBUG) printf("Thread %d Going to Sleep!\n",ThreadId);
      ierr = pthread_cond_wait(&job_true.cond,&job_true.mutex);
    }
    job_true.startJob = PETSC_FALSE;
    job_true.iNumJobThreads--;
    job_true.iNumReadyThreads--;
    iVal = PetscMaxThreads-job_true.iNumReadyThreads-1;
    pthread_mutex_unlock(&job_true.mutex);
    if(job_true.pdata==NULL) {
      iterr = (PetscErrorCode)(long int)job_true.pfunc(job_true.pdata);
    }
    else {
      iterr = (PetscErrorCode)(long int)job_true.pfunc(job_true.pdata[iVal]);
    }
    if(iterr!=0) {
      ithreaderr = 1;
    }
    if (PETSC_PTHREADCLASSES_DEBUG) printf("Thread %d Finished Job\n",ThreadId);
    /* the barrier is necessary BECAUSE: look at job_true.iNumReadyThreads
      what happens if a thread finishes before they all start? BAD!
     what happens if a thread finishes before any else start? BAD! */
    pthread_barrier_wait(job_true.pbarr); /* ensures all threads are finished */
    /* reset job */
    if(PetscThreadGo) {
      pthread_mutex_lock(&job_true.mutex);
      job_true.iNumReadyThreads++;
      if(job_true.iNumReadyThreads==PetscMaxThreads) {
	/* signal the 'main' thread that the job is done! (only done once) */
	ierr = pthread_cond_signal(&main_cond);
      }
    }
  }
  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInitialize_True"
void* PetscThreadInitialize_True(PetscInt N) {
  PetscInt i;
  int status;

  pVal = (int*)malloc(N*sizeof(int));
  /* allocate memory in the heap for the thread structure */
  PetscThreadPoint = (pthread_t*)malloc(N*sizeof(pthread_t));
  BarrPoint = (pthread_barrier_t*)malloc((N+1)*sizeof(pthread_barrier_t)); /* BarrPoint[0] makes no sense, don't use it! */
  job_true.pdata = (void**)malloc(N*sizeof(void*));
  for(i=0; i<N; i++) {
    pVal[i] = i;
    status = pthread_create(&PetscThreadPoint[i],NULL,PetscThreadFunc,&pVal[i]);
    /* error check to ensure proper thread creation */
    status = pthread_barrier_init(&BarrPoint[i+1],NULL,i+1);
    /* should check error */
  }
  if (PETSC_PTHREADCLASSES_DEBUG) printf("Finished True Thread Pool Initialization\n");
  return NULL;
}


#undef __FUNCT__
#define __FUNCT__ "PetscThreadFinalize_True"
PetscErrorCode PetscThreadFinalize_True() {
  int i,ierr;
  void* jstatus;

  PetscFunctionBegin;

  MainJob(FuncFinish,NULL,PetscMaxThreads);  /* set up job and broadcast work */
  /* join the threads */
  for(i=0; i<PetscMaxThreads; i++) {
    ierr = pthread_join(PetscThreadPoint[i],&jstatus);
  }
  free(BarrPoint);
  free(PetscThreadPoint);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MainWait_True"
void MainWait_True() {
  int ierr;
  ierr = pthread_mutex_lock(&job_true.mutex);
  while(job_true.iNumReadyThreads<PetscMaxThreads||job_true.startJob==PETSC_TRUE) {
    ierr = pthread_cond_wait(&main_cond,&job_true.mutex);
  }
  ierr = pthread_mutex_unlock(&job_true.mutex);
}

#undef __FUNCT__
#define __FUNCT__ "MainJob_True"
PetscErrorCode MainJob_True(void* (*pFunc)(void*),void** data,PetscInt n) {
  int ierr;
  PetscErrorCode ijoberr = 0;

  MainWait();
  job_true.pfunc = pFunc;
  job_true.pdata = data;
  job_true.pbarr = &BarrPoint[n];
  job_true.iNumJobThreads = n;
  job_true.startJob = PETSC_TRUE;
  ierr = pthread_cond_broadcast(&job_true.cond);
  if(pFunc!=FuncFinish) {
    MainWait(); /* why wait after? guarantees that job gets done */
  }

  if(ithreaderr) {
    ijoberr = ithreaderr;
  }
  return ijoberr;
}
/**** NO THREAD POOL FUNCTION ****/
#undef __FUNCT__
#define __FUNCT__ "MainJob_Spawn"
PetscErrorCode MainJob_Spawn(void* (*pFunc)(void*),void** data,PetscInt n) {
  PetscErrorCode ijoberr = 0;

  pthread_t* apThread = (pthread_t*)malloc(n*sizeof(pthread_t));
  PetscThreadPoint = apThread; /* point to same place */
  PetscThreadRun(MPI_COMM_WORLD,pFunc,n,apThread,data);
  PetscThreadStop(MPI_COMM_WORLD,n,apThread); /* ensures that all threads are finished with the job */
  free(apThread);

  return ijoberr;
}
/****  ****/
#endif

void* FuncFinish(void* arg) {
  PetscThreadGo = PETSC_FALSE;
  return(0);
}

#endif
