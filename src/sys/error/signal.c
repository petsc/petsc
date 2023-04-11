
/*
      Routines to handle signals the program will receive.
    Usually this will call the error handlers.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"   I*/
#include <signal.h>
#include <stdlib.h> /* for _Exit() */

static PetscClassId SIGNAL_CLASSID = 0;

struct SH {
  PetscClassId classid;
  PetscErrorCode (*handler)(int, void *);
  void      *ctx;
  struct SH *previous;
};
static struct SH *sh        = NULL;
static PetscBool  SignalSet = PETSC_FALSE;

/* Called by MPI_Abort() to suppress user-registered atexit()/on_exit() functions.
   See discussion at https://gitlab.com/petsc/petsc/-/merge_requests/2745.
*/
static void MyExit(void)
{
  _Exit(MPI_ERR_OTHER);
}

/*
    PetscSignalHandler_Private - This is the signal handler called by the system. This calls
             any signal handler set by PETSc or the application code.

   Input Parameters: (depends on system)
.    sig - integer code indicating the type of signal
.    code - ??
.    sigcontext - ??
.    addr - ??

*/
#if defined(PETSC_HAVE_4ARG_SIGNAL_HANDLER)
static void PetscSignalHandler_Private(int sig, int code, struct sigcontext *scp, char *addr)
#else
static void PetscSignalHandler_Private(int sig)
#endif
{
  PetscErrorCode ierr;

  if (!sh || !sh->handler) ierr = PetscSignalHandlerDefault(sig, (void *)0);
  else {
    if (sh->classid != SIGNAL_CLASSID) SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_COR, "Signal object has been corrupted");
    ierr = (*sh->handler)(sig, sh->ctx);
  }
  if (ierr) PETSCABORT(PETSC_COMM_WORLD, PETSC_ERR_COR);
}

/*@
   PetscSignalHandlerDefault - Default signal handler.

   Not Collective

   Input Parameters:
+  sig - signal value
-  ptr - unused pointer

   Level: advanced

   Developer Note:
   This does not call `PetscError()`, handles the entire error process directly

.seealso: `PetscPushSignalHandler()`
@*/
PetscErrorCode PetscSignalHandlerDefault(int sig, void *ptr)
{
  PetscErrorCode ierr;
  const char    *SIGNAME[64];

  if (sig == SIGSEGV) PetscSignalSegvCheckPointerOrMpi();
  SIGNAME[0] = "Unknown signal";
#if !defined(PETSC_MISSING_SIGABRT)
  SIGNAME[SIGABRT] = "Abort";
#endif
#if !defined(PETSC_MISSING_SIGALRM)
  SIGNAME[SIGALRM] = "Alarm";
#endif
#if !defined(PETSC_MISSING_SIGBUS)
  SIGNAME[SIGBUS] = "BUS: Bus Error, possibly illegal memory access";
#endif
#if !defined(PETSC_MISSING_SIGCHLD)
  SIGNAME[SIGCHLD] = "CHLD";
#endif
#if !defined(PETSC_MISSING_SIGCONT)
  SIGNAME[SIGCONT] = "CONT";
#endif
#if !defined(PETSC_MISSING_SIGFPE)
  SIGNAME[SIGFPE] = "FPE: Floating Point Exception,probably divide by zero";
#endif
#if !defined(PETSC_MISSING_SIGHUP)
  SIGNAME[SIGHUP] = "Hang up: Some other process (or the batch system) has told this process to end";
#endif
#if !defined(PETSC_MISSING_SIGILL)
  SIGNAME[SIGILL] = "Illegal instruction: Likely due to memory corruption";
#endif
#if !defined(PETSC_MISSING_SIGINT)
  SIGNAME[SIGINT] = "Interrupt";
#endif
#if !defined(PETSC_MISSING_SIGKILL)
  SIGNAME[SIGKILL] = "Kill: Some other process (or the batch system) has told this process to end";
#endif
#if !defined(PETSC_MISSING_SIGPIPE)
  SIGNAME[SIGPIPE] = "Broken Pipe: Likely while reading or writing to a socket";
#endif
#if !defined(PETSC_MISSING_SIGQUIT)
  SIGNAME[SIGQUIT] = "Quit: Some other process (or the batch system) has told this process to end";
#endif
#if !defined(PETSC_MISSING_SIGSEGV)
  SIGNAME[SIGSEGV] = "SEGV: Segmentation Violation, probably memory access out of range";
#endif
#if !defined(PETSC_MISSING_SIGSYS)
  SIGNAME[SIGSYS] = "SYS";
#endif
#if !defined(PETSC_MISSING_SIGTERM)
  SIGNAME[SIGTERM] = "Terminate: Some process (or the batch system) has told this process to end";
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
  SIGNAME[SIGTRAP] = "TRAP";
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
  SIGNAME[SIGTSTP] = "TSTP";
#endif
#if !defined(PETSC_MISSING_SIGURG)
  SIGNAME[SIGURG] = "URG";
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
  SIGNAME[SIGUSR1] = "User 1";
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
  SIGNAME[SIGUSR2] = "User 2";
#endif

  signal(sig, SIG_DFL);
  ierr = PetscSleep(PetscGlobalRank % 4); /* prevent some jumbling of error messages from different ranks */
  ierr = (*PetscErrorPrintf)("------------------------------------------------------------------------\n");
  if (sig >= 0 && sig <= 20) ierr = (*PetscErrorPrintf)("Caught signal number %d %s\n", sig, SIGNAME[sig]);
  else ierr = (*PetscErrorPrintf)("Caught signal\n");

  ierr = (*PetscErrorPrintf)("Try option -start_in_debugger or -on_error_attach_debugger\n");
  ierr = (*PetscErrorPrintf)("or see https://petsc.org/release/faq/#valgrind and https://petsc.org/release/faq/\n");
#if defined(PETSC_HAVE_CUDA)
  ierr = (*PetscErrorPrintf)("or try https://docs.nvidia.com/cuda/cuda-memcheck/index.html on NVIDIA CUDA systems to find memory corruption errors\n");
#endif
#if PetscDefined(USE_DEBUG)
  #if !PetscDefined(HAVE_THREADSAFETY)
  ierr = (*PetscErrorPrintf)("---------------------  Stack Frames ------------------------------------\n");
  ierr = PetscStackView(PETSC_STDOUT);
  #endif
#else
  ierr = (*PetscErrorPrintf)("configure using --with-debugging=yes, recompile, link, and run \n");
  ierr = (*PetscErrorPrintf)("to get more information on the crash.\n");
#endif
#if !defined(PETSC_MISSING_SIGBUS)
  if (sig == SIGSEGV || sig == SIGBUS) {
#else
  if (sig == SIGSEGV) {
#endif
    PetscBool debug;

    ierr = PetscMallocGetDebug(&debug, NULL, NULL);
    if (debug) ierr = PetscMallocValidate(__LINE__, PETSC_FUNCTION_NAME, __FILE__);
    else ierr = (*PetscErrorPrintf)("Run with -malloc_debug to check if memory corruption is causing the crash.\n");
  }
  atexit(MyExit);
  (void)ierr;
  PETSCABORT(PETSC_COMM_WORLD, PETSC_ERR_SIG);
  return PETSC_SUCCESS;
}

#if !defined(PETSC_SIGNAL_CAST)
  #define PETSC_SIGNAL_CAST
#endif

/*@C
   PetscPushSignalHandler - Catches the usual fatal errors and
   calls a user-provided routine.

   Not Collective

   Input Parameters:
+  routine - routine to call when a signal is received
-  ctx - optional context needed by the routine

  Level: developer

.seealso: `PetscPopSignalHandler()`, `PetscSignalHandlerDefault()`, `PetscPushErrorHandler()`
@*/
PetscErrorCode PetscPushSignalHandler(PetscErrorCode (*routine)(int, void *), void *ctx)
{
  struct SH *newsh;

  PetscFunctionBegin;
  if (!SIGNAL_CLASSID) {
    /* PetscCall(PetscClassIdRegister("Signal",&SIGNAL_CLASSID)); */
    SIGNAL_CLASSID = 19;
  }
  if (!SignalSet && routine) {
    /* Do not catch ABRT, CHLD, KILL */
#if !defined(PETSC_MISSING_SIGALRM)
    /* signal(SIGALRM, PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
#if !defined(PETSC_MISSING_SIGBUS)
    signal(SIGBUS, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGCONT)
    /*signal(SIGCONT, PETSC_SIGNAL_CAST PetscSignalHandler_Private);*/
#endif
#if !defined(PETSC_MISSING_SIGFPE)
    signal(SIGFPE, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGHUP) && defined(PETSC_HAVE_STRUCT_SIGACTION)
    {
      struct sigaction action;
      sigaction(SIGHUP, NULL, &action);
      if (action.sa_handler == SIG_IGN) {
        PetscCall(PetscInfo(NULL, "SIGHUP previously set to ignore, therefore not changing its signal handler\n"));
      } else {
        signal(SIGHUP, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
      }
    }
#endif
#if !defined(PETSC_MISSING_SIGILL)
    signal(SIGILL, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGINT)
    /* signal(SIGINT, PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
#if !defined(PETSC_MISSING_SIGPIPE)
    signal(SIGPIPE, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGQUIT)
    signal(SIGQUIT, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGSEGV)
    signal(SIGSEGV, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGSYS)
    signal(SIGSYS, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGTERM)
  #if !defined(OMPI_MAJOR_VERSION)
    /* OpenMPI may use SIGTERM to close down all its ranks; we don't want to generate many confusing PETSc error messages in that case */
    signal(SIGTERM, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
  #endif
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
    signal(SIGTRAP, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
    /* signal(SIGTSTP,  PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
#if !defined(PETSC_MISSING_SIGURG)
    signal(SIGURG, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
    /* signal(SIGUSR1, PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
    /* signal(SIGUSR2, PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
    SignalSet = PETSC_TRUE;
  }
  if (!routine) {
#if !defined(PETSC_MISSING_SIGALRM)
    /* signal(SIGALRM, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGBUS)
    signal(SIGBUS, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGCONT)
    /* signal(SIGCONT, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGFPE)
    signal(SIGFPE, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGHUP)
    signal(SIGHUP, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGILL)
    signal(SIGILL, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGINT)
    /* signal(SIGINT,  SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGPIPE)
    signal(SIGPIPE, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGQUIT)
    signal(SIGQUIT, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGSEGV)
    signal(SIGSEGV, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGSYS)
    signal(SIGSYS, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGTERM)
    signal(SIGTERM, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
    signal(SIGTRAP, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
    /* signal(SIGTSTP, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGURG)
    signal(SIGURG, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
    /* signal(SIGUSR1, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
    /* signal(SIGUSR2, SIG_DFL); */
#endif
    SignalSet = PETSC_FALSE;
  }
  PetscCall(PetscNew(&newsh));
  if (sh) {
    PetscCheck(sh->classid == SIGNAL_CLASSID, PETSC_COMM_SELF, PETSC_ERR_COR, "Signal object has been corrupted");
    newsh->previous = sh;
  } else newsh->previous = NULL;
  newsh->handler = routine;
  newsh->ctx     = ctx;
  newsh->classid = SIGNAL_CLASSID;
  sh             = newsh;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscPopSignalHandler - Removes the most last signal handler that was pushed.
       If no signal handlers are left on the stack it will remove the PETSc signal handler.
       (That is PETSc will no longer catch signals).

   Not Collective

  Level: developer

.seealso: `PetscPushSignalHandler()`
@*/
PetscErrorCode PetscPopSignalHandler(void)
{
  struct SH *tmp;

  PetscFunctionBegin;
  if (!sh) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(sh->classid == SIGNAL_CLASSID, PETSC_COMM_SELF, PETSC_ERR_COR, "Signal object has been corrupted");

  tmp = sh;
  sh  = sh->previous;
  PetscCall(PetscFree(tmp));
  if (!sh || !sh->handler) {
#if !defined(PETSC_MISSING_SIGALRM)
    /* signal(SIGALRM, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGBUS)
    signal(SIGBUS, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGCONT)
    /* signal(SIGCONT, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGFPE)
    signal(SIGFPE, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGHUP)
    signal(SIGHUP, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGILL)
    signal(SIGILL, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGINT)
    /* signal(SIGINT,  SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGPIPE)
    signal(SIGPIPE, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGQUIT)
    signal(SIGQUIT, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGSEGV)
    signal(SIGSEGV, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGSYS)
    signal(SIGSYS, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGTERM)
    signal(SIGTERM, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
    signal(SIGTRAP, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
    /* signal(SIGTSTP, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGURG)
    signal(SIGURG, SIG_DFL);
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
    /* signal(SIGUSR1, SIG_DFL); */
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
    /* signal(SIGUSR2, SIG_DFL); */
#endif
    SignalSet = PETSC_FALSE;
  } else {
    SignalSet = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
