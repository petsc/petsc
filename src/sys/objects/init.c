/*

   This file defines part of the initialization of PETSc

  This file uses regular malloc and free because it cannot be known
  what malloc is being used until it has already processed the input.
*/
#include <petsc/private/petscimpl.h> /*I  "petscsys.h"   I*/

#if defined(PETSC_HAVE_UNISTD_H)
  #include <unistd.h>
#endif

/* ------------------------Nasty global variables -------------------------------*/
/*
     Indicates if PETSc started up MPI, or it was
   already started before PETSc was initialized.
*/
PetscBool PetscBeganMPI                 = PETSC_FALSE;
PetscBool PetscErrorHandlingInitialized = PETSC_FALSE;
PetscBool PetscInitializeCalled         = PETSC_FALSE;
PetscBool PetscFinalizeCalled           = PETSC_FALSE;

PetscMPIInt PetscGlobalRank = -1;
PetscMPIInt PetscGlobalSize = -1;

#if defined(PETSC_HAVE_KOKKOS)
PetscBool PetscBeganKokkos = PETSC_FALSE;
#endif

#if defined(PETSC_HAVE_NVSHMEM)
PetscBool PetscBeganNvshmem       = PETSC_FALSE;
PetscBool PetscNvshmemInitialized = PETSC_FALSE;
#endif

PetscBool use_gpu_aware_mpi = PetscDefined(HAVE_MPIUNI) ? PETSC_FALSE : PETSC_TRUE;

PetscBool PetscPrintFunctionList = PETSC_FALSE;

#if defined(PETSC_HAVE_COMPLEX)
  #if defined(PETSC_COMPLEX_INSTANTIATE)
template <>
class std::complex<double>; /* instantiate complex template class */
  #endif

/*MC
   PETSC_i - the imaginary number i

   Synopsis:
   #include <petscsys.h>
   PetscComplex PETSC_i;

   Level: beginner

   Note:
   Complex numbers are automatically available if PETSc located a working complex implementation

.seealso: `PetscRealPart()`, `PetscImaginaryPart()`, `PetscRealPartComplex()`, `PetscImaginaryPartComplex()`
M*/
PetscComplex PETSC_i;
MPI_Datatype MPIU___COMPLEX128 = 0;
#endif /* PETSC_HAVE_COMPLEX */
#if defined(PETSC_HAVE_REAL___FLOAT128)
MPI_Datatype MPIU___FLOAT128 = 0;
#endif
#if defined(PETSC_HAVE_REAL___FP16)
MPI_Datatype MPIU___FP16 = 0;
#endif
MPI_Datatype MPIU_2SCALAR    = 0;
MPI_Datatype MPIU_REAL_INT   = 0;
MPI_Datatype MPIU_SCALAR_INT = 0;
#if defined(PETSC_USE_64BIT_INDICES)
MPI_Datatype MPIU_2INT = 0;
#endif
MPI_Datatype MPI_4INT  = 0;
MPI_Datatype MPIU_4INT = 0;
MPI_Datatype MPIU_BOOL;
MPI_Datatype MPIU_ENUM;
MPI_Datatype MPIU_FORTRANADDR;
MPI_Datatype MPIU_SIZE_T;

/*
       Function that is called to display all error messages
*/
PetscErrorCode (*PetscErrorPrintf)(const char[], ...)          = PetscErrorPrintfDefault;
PetscErrorCode (*PetscHelpPrintf)(MPI_Comm, const char[], ...) = PetscHelpPrintfDefault;
PetscErrorCode (*PetscVFPrintf)(FILE *, const char[], va_list) = PetscVFPrintfDefault;

/* ------------------------------------------------------------------------------*/
/*
   Optional file where all PETSc output from various prints is saved
*/
PETSC_INTERN FILE *petsc_history;
FILE              *petsc_history = NULL;

PetscErrorCode PetscOpenHistoryFile(const char filename[], FILE **fd)
{
  PetscMPIInt rank, size;
  char        pfile[PETSC_MAX_PATH_LEN], pname[PETSC_MAX_PATH_LEN], fname[PETSC_MAX_PATH_LEN], date[64];
  char        version[256];

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    char arch[10];

    PetscCall(PetscGetArchType(arch, 10));
    PetscCall(PetscGetDate(date, 64));
    PetscCall(PetscGetVersion(version, 256));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    if (filename) {
      PetscCall(PetscFixFilename(filename, fname));
    } else {
      PetscCall(PetscGetHomeDirectory(pfile, sizeof(pfile)));
      PetscCall(PetscStrlcat(pfile, "/.petschistory", sizeof(pfile)));
      PetscCall(PetscFixFilename(pfile, fname));
    }

    *fd = fopen(fname, "a");
    PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open file: %s", fname);

    PetscCall(PetscFPrintf(PETSC_COMM_SELF, *fd, "----------------------------------------\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, *fd, "%s %s\n", version, date));
    PetscCall(PetscGetProgramName(pname, sizeof(pname)));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, *fd, "%s on a %s, %d proc. with options:\n", pname, arch, size));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, *fd, "----------------------------------------\n"));

    PetscCall(PetscFFlush(*fd));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscCloseHistoryFile(FILE **fd)
{
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    char date[64];
    int  err;

    PetscCall(PetscGetDate(date, sizeof(date)));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, *fd, "----------------------------------------\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, *fd, "Finished at %s\n", date));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, *fd, "----------------------------------------\n"));
    PetscCall(PetscFFlush(*fd));
    err = fclose(*fd);
    PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------------------*/

/*
   This is ugly and probably belongs somewhere else, but I want to
  be able to put a true MPI abort error handler with command line args.

    This is so MPI errors in the debugger will leave all the stack
  frames. The default MP_Abort() cleans up and exits thus providing no useful information
  in the debugger hence we call abort() instead of MPI_Abort().
*/

void Petsc_MPI_AbortOnError(PETSC_UNUSED MPI_Comm *comm, PetscMPIInt *flag, ...)
{
  PetscFunctionBegin;
  PetscCallContinue((*PetscErrorPrintf)("MPI error %d\n", *flag));
  abort();
}

void Petsc_MPI_DebuggerOnError(MPI_Comm *comm, PetscMPIInt *flag, ...)
{
  PetscFunctionBegin;
  PetscCallContinue((*PetscErrorPrintf)("MPI error %d\n", *flag));
  if (PetscAttachDebugger()) PETSCABORT(*comm, (PetscErrorCode)*flag); /* hopeless so get out */
}

/*@C
   PetscEnd - Calls `PetscFinalize()` and then ends the program. This is useful if one
     wishes a clean exit somewhere deep in the program.

   Collective on `PETSC_COMM_WORLD`

   Options Database Keys are the same as for `PetscFinalize()`

   Level: advanced

   Note:
   See `PetscInitialize()` for more general runtime options.

.seealso: `PetscInitialize()`, `PetscOptionsView()`, `PetscMallocDump()`, `PetscMPIDump()`, `PetscFinalize()`
@*/
PetscErrorCode PetscEnd(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFinalize());
  exit(0);
  return PETSC_SUCCESS;
}

PetscBool                   PetscOptionsPublish = PETSC_FALSE;
PETSC_INTERN PetscErrorCode PetscSetUseHBWMalloc_Private(void);
PETSC_INTERN PetscBool      petscsetmallocvisited;
static char                 emacsmachinename[256];

PetscErrorCode (*PetscExternalVersionFunction)(MPI_Comm) = NULL;
PetscErrorCode (*PetscExternalHelpFunction)(MPI_Comm)    = NULL;

#if PetscDefined(USE_LOG)
  #include <petscviewer.h>
#endif

/*@C
   PetscSetHelpVersionFunctions - Sets functions that print help and version information
   before the PETSc help and version information is printed. Must call BEFORE `PetscInitialize()`.
   This routine enables a "higher-level" package that uses PETSc to print its messages first.

   Input Parameters:
+  help - the help function (may be NULL)
-  version - the version function (may be NULL)

   Level: developer

@*/
PetscErrorCode PetscSetHelpVersionFunctions(PetscErrorCode (*help)(MPI_Comm), PetscErrorCode (*version)(MPI_Comm))
{
  PetscFunctionBegin;
  PetscExternalHelpFunction    = help;
  PetscExternalVersionFunction = version;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_LOG)
PETSC_INTERN PetscBool PetscObjectsLog;
#endif

PETSC_INTERN PetscErrorCode PetscOptionsCheckInitial_Private(const char help[])
{
  char        string[64];
  MPI_Comm    comm = PETSC_COMM_WORLD;
  PetscBool   flg1 = PETSC_FALSE, flg2 = PETSC_FALSE, flag, hasHelp;
  PetscBool   checkstack = PETSC_FALSE;
  PetscReal   si;
  PetscInt    intensity;
  int         i;
  PetscMPIInt rank;
  char        version[256];
#if defined(PETSC_USE_LOG)
  char              mname[PETSC_MAX_PATH_LEN];
  PetscViewerFormat format;
  PetscBool         flg4 = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)) checkstack = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-checkstack", &checkstack, NULL));
  PetscCall(PetscStackSetCheck(checkstack));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-checkfunctionlist", &PetscPrintFunctionList, NULL));

#if !defined(PETSC_HAVE_THREADSAFETY)
  if (!(PETSC_RUNNING_ON_VALGRIND)) {
    /*
      Setup the memory management; support for tracing malloc() usage
    */
    PetscBool mdebug = PETSC_FALSE, eachcall = PETSC_FALSE, initializenan = PETSC_FALSE, mlog = PETSC_FALSE;
    PetscBool flg3 = PETSC_FALSE;

    if (PetscDefined(USE_DEBUG)) {
      mdebug        = PETSC_TRUE;
      initializenan = PETSC_TRUE;
      PetscCall(PetscOptionsHasName(NULL, NULL, "-malloc_test", &flg1));
    } else {
      /* don't warn about unused option */
      PetscCall(PetscOptionsHasName(NULL, NULL, "-malloc_test", &flg1));
      flg1 = PETSC_FALSE;
    }
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_debug", &flg2, &flg3));
    if (flg1 || flg2) {
      mdebug        = PETSC_TRUE;
      eachcall      = PETSC_TRUE;
      initializenan = PETSC_TRUE;
    } else if (flg3 && !flg2) {
      mdebug        = PETSC_FALSE;
      eachcall      = PETSC_FALSE;
      initializenan = PETSC_FALSE;
    }

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_requested_size", &flg1, &flg2));
    if (flg2) PetscCall(PetscMallocLogRequestedSizeSet(flg1));

    PetscCall(PetscOptionsHasName(NULL, NULL, "-malloc_view", &mlog));
    if (mlog) mdebug = PETSC_TRUE;
    /* the next line is deprecated */
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc", &mdebug, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_dump", &mdebug, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-log_view_memory", &mdebug, NULL));
    if (mdebug) PetscCall(PetscMallocSetDebug(eachcall, initializenan));
    if (mlog) {
      PetscReal logthreshold = 0;
      PetscCall(PetscOptionsGetReal(NULL, NULL, "-malloc_view_threshold", &logthreshold, NULL));
      PetscCall(PetscMallocViewSet(logthreshold));
    }
  #if defined(PETSC_USE_LOG)
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-log_view_memory", &PetscLogMemory, NULL));
  #endif
  }

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_coalesce", &flg1, &flg2));
  if (flg2) PetscCall(PetscMallocSetCoalesce(flg1));
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_hbw", &flg1, NULL));
  /* ignore this option if malloc is already set */
  if (flg1 && !petscsetmallocvisited) PetscCall(PetscSetUseHBWMalloc_Private());

  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-malloc_info", &flg1, NULL));
  if (!flg1) {
    flg1 = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-memory_view", &flg1, NULL));
  }
  if (flg1) PetscCall(PetscMemorySetGetMaximumUsage());
#endif

#if defined(PETSC_USE_LOG)
  PetscCall(PetscOptionsHasName(NULL, NULL, "-objects_dump", &PetscObjectsLog));
#endif

  /*
      Set the display variable for graphics
  */
  PetscCall(PetscSetDisplay());

  /*
     Print main application help message
  */
  PetscCall(PetscOptionsHasHelp(NULL, &hasHelp));
  if (help && hasHelp) {
    PetscCall(PetscPrintf(comm, "%s", help));
    PetscCall(PetscPrintf(comm, "----------------------------------------\n"));
  }

  /*
      Print the PETSc version information
  */
  PetscCall(PetscOptionsHasName(NULL, NULL, "-version", &flg1));
  if (flg1 || hasHelp) {
    /*
       Print "higher-level" package version message
    */
    if (PetscExternalVersionFunction) PetscCall((*PetscExternalVersionFunction)(comm));

    PetscCall(PetscGetVersion(version, 256));
    PetscCall((*PetscHelpPrintf)(comm, "%s\n", version));
    PetscCall((*PetscHelpPrintf)(comm, "%s", PETSC_AUTHOR_INFO));
    PetscCall((*PetscHelpPrintf)(comm, "See https://petsc.org/release/changes for recent updates.\n"));
    PetscCall((*PetscHelpPrintf)(comm, "See https://petsc.org/release/faq for problems.\n"));
    PetscCall((*PetscHelpPrintf)(comm, "See https://petsc.org/release/manualpages for help. \n"));
    PetscCall((*PetscHelpPrintf)(comm, "Libraries linked from %s\n", PETSC_LIB_DIR));
    PetscCall((*PetscHelpPrintf)(comm, "----------------------------------------\n"));
  }

  /*
       Print "higher-level" package help message
  */
  if (hasHelp) {
    PetscBool hasHelpIntro;

    if (PetscExternalHelpFunction) PetscCall((*PetscExternalHelpFunction)(comm));
    PetscCall(PetscOptionsHasHelpIntro_Internal(NULL, &hasHelpIntro));
    if (hasHelpIntro) {
      PetscCall(PetscOptionsDestroyDefault());
      PetscCall(PetscFreeMPIResources());
      PetscCallMPI(MPI_Finalize());
      exit(0);
    }
  }

  /*
      Setup the error handling
  */
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-on_error_abort", &flg1, NULL));
  if (flg1) {
    PetscCallMPI(MPI_Comm_set_errhandler(comm, MPI_ERRORS_ARE_FATAL));
    PetscCall(PetscPushErrorHandler(PetscAbortErrorHandler, NULL));
  }
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-on_error_mpiabort", &flg1, NULL));
  if (flg1) PetscCall(PetscPushErrorHandler(PetscMPIAbortErrorHandler, NULL));
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mpi_return_on_error", &flg1, NULL));
  if (flg1) PetscCallMPI(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_signal_handler", &flg1, NULL));
  if (!flg1) PetscCall(PetscPushSignalHandler(PetscSignalHandlerDefault, (void *)0));

  /*
      Setup debugger information
  */
  PetscCall(PetscSetDefaultDebugger());
  PetscCall(PetscOptionsGetString(NULL, NULL, "-on_error_attach_debugger", string, sizeof(string), &flg1));
  if (flg1) {
    MPI_Errhandler err_handler;

    PetscCall(PetscSetDebuggerFromString(string));
    PetscCallMPI(MPI_Comm_create_errhandler(Petsc_MPI_DebuggerOnError, &err_handler));
    PetscCallMPI(MPI_Comm_set_errhandler(comm, err_handler));
    PetscCall(PetscPushErrorHandler(PetscAttachDebuggerErrorHandler, NULL));
  }
  PetscCall(PetscOptionsGetString(NULL, NULL, "-debug_terminal", string, sizeof(string), &flg1));
  if (flg1) PetscCall(PetscSetDebugTerminal(string));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-start_in_debugger", string, sizeof(string), &flg1));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-stop_for_debugger", string, sizeof(string), &flg2));
  if (flg1 || flg2) {
    PetscMPIInt    size;
    PetscInt       lsize, *ranks;
    MPI_Errhandler err_handler;
    /*
       we have to make sure that all processors have opened
       connections to all other processors, otherwise once the
       debugger has stated it is likely to receive a SIGUSR1
       and kill the program.
    */
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (size > 2) {
      PetscMPIInt dummy = 0;
      MPI_Status  status;
      for (i = 0; i < size; i++) {
        if (rank != i) PetscCallMPI(MPI_Send(&dummy, 1, MPI_INT, i, 109, comm));
      }
      for (i = 0; i < size; i++) {
        if (rank != i) PetscCallMPI(MPI_Recv(&dummy, 1, MPI_INT, i, 109, comm, &status));
      }
    }
    /* check if this processor node should be in debugger */
    PetscCall(PetscMalloc1(size, &ranks));
    lsize = size;
    /* Deprecated in 3.14 */
    PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-debugger_nodes", ranks, &lsize, &flag));
    if (flag) {
      const char *const quietopt = "-options_suppress_deprecated_warnings";
      char              msg[4096];
      PetscBool         quiet = PETSC_FALSE;

      PetscCall(PetscOptionsGetBool(NULL, NULL, quietopt, &quiet, NULL));
      if (!quiet) {
        PetscCall(PetscStrncpy(msg, "** PETSc DEPRECATION WARNING ** : the option ", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, "-debugger_nodes", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, " is deprecated as of version ", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, "3.14", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, " and will be removed in a future release.", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, " Please use the option ", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, "-debugger_ranks", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, " instead.", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, " (Silence this warning with ", sizeof(msg)));
        PetscCall(PetscStrlcat(msg, quietopt, sizeof(msg)));
        PetscCall(PetscStrlcat(msg, ")\n", sizeof(msg)));
        PetscCall(PetscPrintf(comm, "%s", msg));
      }
    } else {
      lsize = size;
      PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-debugger_ranks", ranks, &lsize, &flag));
    }
    if (flag) {
      for (i = 0; i < lsize; i++) {
        if (ranks[i] == rank) {
          flag = PETSC_FALSE;
          break;
        }
      }
    }
    if (!flag) {
      PetscCall(PetscSetDebuggerFromString(string));
      PetscCall(PetscPushErrorHandler(PetscAbortErrorHandler, NULL));
      if (flg1) {
        PetscCall(PetscAttachDebugger());
      } else {
        PetscCall(PetscStopForDebugger());
      }
      PetscCallMPI(MPI_Comm_create_errhandler(Petsc_MPI_AbortOnError, &err_handler));
      PetscCallMPI(MPI_Comm_set_errhandler(comm, err_handler));
    } else {
      PetscCall(PetscWaitOnError());
    }
    PetscCall(PetscFree(ranks));
  }

  PetscCall(PetscOptionsGetString(NULL, NULL, "-on_error_emacs", emacsmachinename, sizeof(emacsmachinename), &flg1));
  if (flg1 && rank == 0) PetscCall(PetscPushErrorHandler(PetscEmacsClientErrorHandler, emacsmachinename));

    /*
        Setup profiling and logging
  */
#if defined(PETSC_USE_INFO)
  {
    PetscCall(PetscInfoSetFromOptions(NULL));
  }
#endif
  PetscCall(PetscDetermineInitialFPTrap());
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-fp_trap", &flg1, &flag));
  if (flag) PetscCall(PetscSetFPTrap(flg1 ? PETSC_FP_TRAP_ON : PETSC_FP_TRAP_OFF));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-check_pointer_intensity", &intensity, &flag));
  if (flag) PetscCall(PetscCheckPointerSetIntensity(intensity));
#if defined(PETSC_USE_LOG)
  mname[0] = 0;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-history", mname, sizeof(mname), &flg1));
  if (flg1) {
    if (mname[0]) {
      PetscCall(PetscOpenHistoryFile(mname, &petsc_history));
    } else {
      PetscCall(PetscOpenHistoryFile(NULL, &petsc_history));
    }
  }

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-log_sync", &PetscLogSyncOn, NULL));

  PetscBool flg3 = PETSC_FALSE;
  #if defined(PETSC_HAVE_MPE)
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-log_mpe", &flg1));
  if (flg1) PetscCall(PetscLogMPEBegin());
  #endif
  flg1 = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-log_all", &flg1, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-log_summary", &flg3));
  if (flg1) PetscCall(PetscLogAllBegin());
  else if (flg3) PetscCall(PetscLogDefaultBegin());

  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_trace", mname, sizeof(mname), &flg1));
  if (flg1) {
    char  name[PETSC_MAX_PATH_LEN], fname[PETSC_MAX_PATH_LEN];
    FILE *file;
    if (mname[0]) {
      PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%s.%d", mname, rank));
      PetscCall(PetscFixFilename(name, fname));
      file = fopen(fname, "w");
      PetscCheck(file, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to open trace file: %s", fname);
    } else file = PETSC_STDOUT;
    PetscCall(PetscLogTraceBegin(file));
  }

  PetscCall(PetscOptionsGetViewer(comm, NULL, NULL, "-log_view", NULL, &format, &flg4));
  if (flg4) {
    if (format == PETSC_VIEWER_ASCII_XML || format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
      PetscCall(PetscLogNestedBegin());
    } else {
      PetscCall(PetscLogDefaultBegin());
    }
  }
  if (flg4 && (format == PETSC_VIEWER_ASCII_XML || format == PETSC_VIEWER_ASCII_FLAMEGRAPH)) {
    PetscReal threshold = PetscRealConstant(0.01);
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-log_threshold", &threshold, &flg1));
    if (flg1) PetscCall(PetscLogSetThreshold((PetscLogDouble)threshold, NULL));
  }
#endif

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-saws_options", &PetscOptionsPublish, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_gpu_aware_mpi", &use_gpu_aware_mpi, &flg1));
  if (!flg1) PetscCall(PetscOptionsGetBool(NULL, NULL, "-sf_use_gpu_aware_mpi", &use_gpu_aware_mpi, &flg1)); // an alias option

  /*
       Print basic help message
  */
  if (hasHelp) {
    PetscCall((*PetscHelpPrintf)(comm, "Options for all PETSc programs:\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -version: prints PETSc version\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -help intro: prints example description and PETSc version, and exits\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -help: prints example description, PETSc version, and available options for used routines\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -on_error_abort: cause an abort when an error is detected. Useful \n "));
    PetscCall((*PetscHelpPrintf)(comm, "       only when run in the debugger\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -on_error_attach_debugger [gdb,dbx,xxgdb,ups,noxterm]\n"));
    PetscCall((*PetscHelpPrintf)(comm, "       start the debugger in new xterm\n"));
    PetscCall((*PetscHelpPrintf)(comm, "       unless noxterm is given\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -start_in_debugger [gdb,dbx,xxgdb,ups,noxterm]\n"));
    PetscCall((*PetscHelpPrintf)(comm, "       start all processes in the debugger\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -on_error_emacs <machinename>\n"));
    PetscCall((*PetscHelpPrintf)(comm, "    emacs jumps to error file\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -debugger_ranks [n1,n2,..] Ranks to start in debugger\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -debugger_pause [m] : delay (in seconds) to attach debugger\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -stop_for_debugger : prints message on how to attach debugger manually\n"));
    PetscCall((*PetscHelpPrintf)(comm, "                      waits the delay for you to attach\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -display display: Location where X window graphics and debuggers are displayed\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -no_signal_handler: do not trap error signals\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -mpi_return_on_error: MPI returns error code, rather than abort on internal error\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -fp_trap: stop on floating point exceptions\n"));
    PetscCall((*PetscHelpPrintf)(comm, "           note on IBM RS6000 this slows run greatly\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -malloc_dump <optional filename>: dump list of unfreed memory at conclusion\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -malloc: use PETSc error checking malloc (deprecated, use -malloc_debug)\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -malloc no: don't use PETSc error checking malloc (deprecated, use -malloc_debug no)\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -malloc_info: prints total memory usage\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -malloc_view <optional filename>: keeps log of all memory allocations, displays in PetscFinalize()\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -malloc_debug <true or false>: enables or disables extended checking for memory corruption\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -options_view: dump list of options inputted\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -options_left: dump list of unused options\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -options_left no: don't dump list of unused options\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -tmp tmpdir: alternative /tmp directory\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -shared_tmp: tmp directory is shared by all processors\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -not_shared_tmp: each processor has separate tmp directory\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -memory_view: print memory usage at end of run\n"));
#if defined(PETSC_USE_LOG)
    PetscCall((*PetscHelpPrintf)(comm, " -get_total_flops: total flops over all processors\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -log_view [:filename:[format]]: logging objects and events\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -log_trace [filename]: prints trace of all PETSc calls\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -log_exclude <list,of,classnames>: exclude given classes from logging\n"));
  #if defined(PETSC_HAVE_DEVICE)
    PetscCall((*PetscHelpPrintf)(comm, " -log_view_gpu_time: log the GPU time for each and event\n"));
  #endif
  #if defined(PETSC_HAVE_MPE)
    PetscCall((*PetscHelpPrintf)(comm, " -log_mpe: Also create logfile viewable through Jumpshot\n"));
  #endif
#endif
#if defined(PETSC_USE_INFO)
    PetscCall((*PetscHelpPrintf)(comm, " -info [filename][:[~]<list,of,classnames>[:[~]self]]: print verbose information\n"));
#endif
    PetscCall((*PetscHelpPrintf)(comm, " -options_file <file>: reads options from file\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -options_monitor: monitor options to standard output, including that set previously e.g. in option files\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -options_monitor_cancel: cancels all hardwired option monitors\n"));
    PetscCall((*PetscHelpPrintf)(comm, " -petsc_sleep n: sleeps n seconds before running program\n"));
  }

#if defined(PETSC_HAVE_POPEN)
  {
    char machine[128];
    PetscCall(PetscOptionsGetString(NULL, NULL, "-popen_machine", machine, sizeof(machine), &flg1));
    if (flg1) PetscCall(PetscPOpenSetMachine(machine));
  }
#endif

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-petsc_sleep", &si, &flg1));
  if (flg1) PetscCall(PetscSleep(si));
  PetscFunctionReturn(PETSC_SUCCESS);
}
