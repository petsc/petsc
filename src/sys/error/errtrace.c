#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for fileno() */
#include <petscsys.h>        /*I "petscsys.h" I*/
#include <petsc/private/petscimpl.h>
#include <petscconfiginfo.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

/*@C
   PetscIgnoreErrorHandler - Deprecated, use PetscReturnErrorHandler(). Ignores the error, allows program to continue as if error did not occure

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
-  ctx - error handler context

   Level: developer

   Notes:
   Most users need not directly employ this routine and the other error
   handlers, but can instead use the simplified interface SETERRQ, which has
   the calling sequence
$     SETERRQ(comm,number,p,mess)

.seealso: `PetscReturnErrorHandler()`
 @*/
PetscErrorCode  PetscIgnoreErrorHandler(MPI_Comm comm,int line,const char *fun,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(n);
}

/* ---------------------------------------------------------------------------------------*/

static char      arch[128],hostname[128],username[128],pname[PETSC_MAX_PATH_LEN],date[128];
static PetscBool PetscErrorPrintfInitializeCalled = PETSC_FALSE;
static char      version[256];

/*
   Initializes arch, hostname, username, date so that system calls do NOT need
   to be made during the error handler.
*/
PetscErrorCode  PetscErrorPrintfInitialize(void)
{
  PetscBool      use_stdout = PETSC_FALSE,use_none = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscGetArchType(arch,sizeof(arch)));
  PetscCall(PetscGetHostName(hostname,sizeof(hostname)));
  PetscCall(PetscGetUserName(username,sizeof(username)));
  PetscCall(PetscGetProgramName(pname,sizeof(pname)));
  PetscCall(PetscGetDate(date,sizeof(date)));
  PetscCall(PetscGetVersion(version,sizeof(version)));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-error_output_stdout",&use_stdout,NULL));
  if (use_stdout) PETSC_STDERR = PETSC_STDOUT;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-error_output_none",&use_none,NULL));
  if (use_none) PetscErrorPrintf = PetscErrorPrintfNone;
  PetscErrorPrintfInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscErrorPrintfNone(const char format[],...)
{
  return 0;
}

PetscErrorCode  PetscErrorPrintfDefault(const char format[],...)
{
  va_list          Argp;
  static PetscBool PetscErrorPrintfCalled = PETSC_FALSE;

  /*
      This function does not call PetscFunctionBegin and PetscFunctionReturn() because
    it may be called by PetscStackView().

      This function does not do error checking because it is called by the error handlers.
  */

  if (!PetscErrorPrintfCalled) {
    PetscErrorPrintfCalled = PETSC_TRUE;

    /*
        On the SGI machines and Cray T3E, if errors are generated  "simultaneously" by
      different processors, the messages are printed all jumbled up; to try to
      prevent this we have each processor wait based on their rank
    */
#if defined(PETSC_CAN_SLEEP_AFTER_ERROR)
    {
      PetscMPIInt rank;
      if (PetscGlobalRank > 8) rank = 8;
      else rank = PetscGlobalRank;
      PetscSleep((PetscReal)rank);
    }
#endif
  }

  PetscFPrintf(PETSC_COMM_SELF,PETSC_STDERR,"[%d]PETSC ERROR: ",PetscGlobalRank);
  va_start(Argp,format);
  (*PetscVFPrintf)(PETSC_STDERR,format,Argp);
  va_end(Argp);
  return 0;
}

/*
   On some systems when the stderr is nested through several levels of shell script
   before being passed to a file the isatty() falsely returns true resulting in
   the screen highlight variables being passed through the test harness. Therefore
   simply do not highlight when the PETSC_STDERR is PETSC_STDOUT.
*/
static void PetscErrorPrintfHilight(void)
{
#if defined(PETSC_HAVE_UNISTD_H) && defined(PETSC_USE_ISATTY)
  if (PetscErrorPrintf == PetscErrorPrintfDefault && PETSC_STDERR != PETSC_STDOUT) {
    if (isatty(fileno(PETSC_STDERR))) fprintf(PETSC_STDERR,"\033[1;31m");
  }
#endif
}

static void PetscErrorPrintfNormal(void)
{
#if defined(PETSC_HAVE_UNISTD_H) && defined(PETSC_USE_ISATTY)
  if (PetscErrorPrintf == PetscErrorPrintfDefault && PETSC_STDERR != PETSC_STDOUT) {
    if (isatty(fileno(PETSC_STDERR))) fprintf(PETSC_STDERR,"\033[0;39m\033[0;49m");
  }
#endif
}

PETSC_EXTERN PetscErrorCode  PetscOptionsViewError(void);

/*@C

   PetscTraceBackErrorHandler - Default error handler routine that generates
   a traceback on error detection.

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - PETSC_ERROR_INITIAL if this is the first call the error handler, otherwise PETSC_ERROR_REPEAT
-  ctx - error handler context

  Options Database:
+  -error_output_stdout - output the error messages to stdout instead of the default stderr
-  -error_output_none - do not output the error messages

   Notes:
   Most users need not directly employ this routine and the other error
   handlers, but can instead use the simplified interface SETERRQ, which has
   the calling sequence
$     SETERRQ(comm,number,n,mess)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.

   Level: developer

.seealso: `PetscError()`, `PetscPushErrorHandler()`, `PetscPopErrorHandler()`, `PetscAttachDebuggerErrorHandler()`,
          `PetscAbortErrorHandler()`, `PetscMPIAbortErrorHandler()`, `PetscReturnErrorHandler()`, `PetscEmacsClientErrorHandler()`
 @*/
PetscErrorCode  PetscTraceBackErrorHandler(MPI_Comm comm,int line,const char *fun,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscLogDouble mem,rss;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE;
  PetscMPIInt    rank = 0;

  if (comm != PETSC_COMM_SELF) MPI_Comm_rank(comm,&rank);

  if (rank == 0 && (!PetscCIEnabledPortableErrorOutput || PetscGlobalRank == 0)) {
    PetscBool  ismain;
    static int cnt = 1;

    if (cnt == 1) {
      PetscErrorPrintfHilight();
      (*PetscErrorPrintf)("--------------------- Error Message --------------------------------------------------------------\n");
      PetscErrorPrintfNormal();
      if (n == PETSC_ERR_MEM) {
        (*PetscErrorPrintf)("Out of memory. This could be due to allocating\n");
        (*PetscErrorPrintf)("too large an object or bleeding by not properly\n");
        (*PetscErrorPrintf)("destroying unneeded objects.\n");
        PetscMallocGetCurrentUsage(&mem);
        PetscMemoryGetCurrentUsage(&rss);
        PetscOptionsGetBool(NULL,NULL,"-malloc_dump",&flg1,NULL);
        PetscOptionsGetBool(NULL,NULL,"-malloc_view",&flg2,NULL);
        PetscOptionsHasName(NULL,NULL,"-malloc_view_threshold",&flg3);
        if (flg2 || flg3) PetscMallocView(stdout);
        else {
          (*PetscErrorPrintf)("Memory allocated %.0f Memory used by process %.0f\n",mem,rss);
          if (flg1) PetscMallocDump(stdout);
          else (*PetscErrorPrintf)("Try running with -malloc_dump or -malloc_view for info.\n");
        }
      } else {
        const char *text;
        PetscErrorMessage(n,&text,NULL);
        if (text) (*PetscErrorPrintf)("%s\n",text);
      }
      if (mess) (*PetscErrorPrintf)("%s\n",mess);
      (*PetscErrorPrintf)("See https://petsc.org/release/faq/ for trouble shooting.\n");
      if (!PetscCIEnabledPortableErrorOutput) {
        (*PetscErrorPrintf)("%s\n",version);
        if (PetscErrorPrintfInitializeCalled) (*PetscErrorPrintf)("%s on a %s named %s by %s %s\n",pname,arch,hostname,username,date);
        (*PetscErrorPrintf)("Configure options %s\n",petscconfigureoptions);
      }
    }
    /* print line of stack trace */
    if (fun) (*PetscErrorPrintf)("#%d %s() at %s:%d\n",cnt++,fun,PetscCIFilename(file),PetscCILinenumber(line));
    else if (file) (*PetscErrorPrintf)("#%d %s:%d\n",cnt++,PetscCIFilename(file),PetscCILinenumber(line));
    if (fun) {
      PetscStrncmp(fun,"main",4,&ismain);
      if (ismain) {
        if ((n <= PETSC_ERR_MIN_VALUE) || (n >= PETSC_ERR_MAX_VALUE)) {
          (*PetscErrorPrintf)("Reached the main program with an out-of-range error code %d. This should never happen\n",n);
        }
        PetscOptionsViewError();
        PetscErrorPrintfHilight();
        (*PetscErrorPrintf)("----------------End of Error Message -------send entire error message to petsc-maint@mcs.anl.gov----------\n");
        PetscErrorPrintfNormal();
      }
    }
  } else {
    /* do not print error messages since process 0 will print them, sleep before aborting so will not accidentally kill process 0*/
    PetscSleep(10.0);
    exit(0);
  }
  return n;
}
