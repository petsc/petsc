
/*
      Routines to handle signals the program will receive. 
    Usually this will call the error handlers.
*/
#include <petscsys.h>             /*I   "petscsys.h"   I*/
#include <signal.h>

static PetscClassId SIGNAL_CLASSID = 0;

struct SH {
  PetscClassId   classid;
  PetscErrorCode (*handler)(int,void *);
  void           *ctx;
  struct SH*     previous;
};
static struct SH* sh        = 0;
static PetscBool  SignalSet = PETSC_FALSE;



EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscSignalHandler_Private"
/*
    PetscSignalHandler_Private - This is the signal handler called by the system. This calls 
             any signal handler set by PETSc or the application code.
 
   Input Parameters: (depends on system)
.    sig - integer code indicating the type of signal
.    code - ??
.    sigcontext - ??
.    addr - ??

    Note: this is declared extern "C" because it is passed to the system routine signal()
          which is an extern "C" routine. The Solaris 2.7 OS compilers require that this be
          extern "C".

*/
#if defined(PETSC_HAVE_4ARG_SIGNAL_HANDLER)
static void PetscSignalHandler_Private(int sig,int code,struct sigcontext * scp,char *addr)
#else
static void PetscSignalHandler_Private(int sig)
#endif
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sh || !sh->handler) {
    ierr = PetscDefaultSignalHandler(sig,(void*)0);
  } else{
    if (sh->classid != SIGNAL_CLASSID) SETERRABORT(PETSC_COMM_WORLD,PETSC_ERR_COR,"Signal object has been corrupted");
    ierr = (*sh->handler)(sig,sh->ctx);
  }
  if (ierr) MPI_Abort(PETSC_COMM_WORLD,0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscDefaultSignalHandler"
/*@
   PetscDefaultSignalHandler - Default signal handler.

   Not Collective

   Level: advanced

   Input Parameters:
+  sig - signal value
-  ptr - unused pointer

   Concepts: signal handler^default

@*/
PetscErrorCode  PetscDefaultSignalHandler(int sig,void *ptr)
{
  PetscErrorCode ierr;
  const char     *SIGNAME[64];

  PetscFunctionBegin;
  SIGNAME[0]       = "Unknown signal";
#if !defined(PETSC_MISSING_SIGABRT)
  SIGNAME[SIGABRT] = "Abort";
#endif
#if !defined(PETSC_MISSING_SIGALRM)
  SIGNAME[SIGALRM] = "Alarm"; 
#endif
#if !defined(PETSC_MISSING_SIGBUS)
  SIGNAME[SIGBUS]  = "BUS: Bus Error, possibly illegal memory access";
#endif
#if !defined(PETSC_MISSING_SIGCHLD)
  SIGNAME[SIGCHLD] = "CHLD";
#endif
#if !defined(PETSC_MISSING_SIGCONT)
  SIGNAME[SIGCONT] = "CONT";
#endif
#if !defined(PETSC_MISSING_SIGFPE)
  SIGNAME[SIGFPE]  = "FPE: Floating Point Exception,probably divide by zero";
#endif
#if !defined(PETSC_MISSING_SIGHUP)
  SIGNAME[SIGHUP]  = "Hang up: Some other process (or the batch system) has told this process to end";
#endif
#if !defined(PETSC_MISSING_SIGILL)
  SIGNAME[SIGILL]  = "Illegal instruction: Likely due to memory corruption";
#endif
#if !defined(PETSC_MISSING_SIGINT)
  SIGNAME[SIGINT]  = "Interrupt"; 
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
  SIGNAME[SIGSYS]  = "SYS";
#endif
#if !defined(PETSC_MISSING_SIGTERM)
  SIGNAME[SIGTERM] = "Terminate: Somet process (or the batch system) has told this process to end";
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
  SIGNAME[SIGTRAP] = "TRAP";
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
  SIGNAME[SIGTSTP] = "TSTP";
#endif
#if !defined(PETSC_MISSING_SIGURG)
  SIGNAME[SIGURG]  = "URG";
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
  SIGNAME[SIGUSR1] = "User 1";
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
  SIGNAME[SIGUSR2] = "User 2";
#endif

  signal(sig,SIG_DFL);
    (*PetscErrorPrintf)("------------------------------------------------------------------------\n");
  if (sig >= 0 && sig <= 20) {
    (*PetscErrorPrintf)("Caught signal number %d %s\n",sig,SIGNAME[sig]);
  } else {
    (*PetscErrorPrintf)("Caught signal\n");
  }
  (*PetscErrorPrintf)("Try option -start_in_debugger or -on_error_attach_debugger\n");
  (*PetscErrorPrintf)("or see http://www.mcs.anl.gov/petsc/documentation/faq.html#valgrind");
  (*PetscErrorPrintf)("or try http://valgrind.org on GNU/linux and Apple Mac OS X to find memory corruption errors\n");
#if defined(PETSC_USE_DEBUG)
  if (!PetscStackActive) {
    (*PetscErrorPrintf)("  or try option -log_stack\n");
  } else {
    PetscStackPop;  /* remove stack frames for error handlers */
    PetscStackPop;
    (*PetscErrorPrintf)("likely location of problem given in stack below\n");
    (*PetscErrorPrintf)("---------------------  Stack Frames ------------------------------------\n");
    PetscStackView(PETSC_VIEWER_STDOUT_SELF);
  }
#endif
#if !defined(PETSC_USE_DEBUG)
  (*PetscErrorPrintf)("configure using --with-debugging=yes, recompile, link, and run \n");
  (*PetscErrorPrintf)("to get more information on the crash.\n");
#endif
  ierr =  PetscError(PETSC_COMM_SELF,0,"User provided function"," unknown file","unknown directory",PETSC_ERR_SIG,PETSC_ERROR_INITIAL,PETSC_NULL);
  MPI_Abort(PETSC_COMM_WORLD,(int)ierr);
  PetscFunctionReturn(0);
}

#if !defined(PETSC_SIGNAL_CAST)
#define PETSC_SIGNAL_CAST
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscPushSignalHandler"
/*@C
   PetscPushSignalHandler - Catches the usual fatal errors and 
   calls a user-provided routine.

   Not Collective

    Input Parameter:
+  routine - routine to call when a signal is received
-  ctx - optional context needed by the routine

  Level: developer

   Concepts: signal handler^setting

.seealso: PetscPopSignalHandler(), PetscDefaultSignalHandler()

@*/
PetscErrorCode  PetscPushSignalHandler(PetscErrorCode (*routine)(int,void*),void* ctx)
{
  struct  SH     *newsh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!SIGNAL_CLASSID) {
    /* ierr = PetscClassIdRegister("Signal",&SIGNAL_CLASSID);CHKERRQ(ierr); */
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
    signal(SIGFPE,  PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGHUP)
    signal(SIGHUP, PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGILL)
    signal(SIGILL,  PETSC_SIGNAL_CAST PetscSignalHandler_Private);
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
    signal(SIGSYS,  PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGTERM)
    signal(SIGTERM,  PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
    signal(SIGTRAP,  PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
    /* signal(SIGTSTP,  PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
#if !defined(PETSC_MISSING_SIGURG)
    signal(SIGURG,  PETSC_SIGNAL_CAST PetscSignalHandler_Private);
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
    /*    signal(SIGUSR1, PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
    /* signal(SIGUSR2, PETSC_SIGNAL_CAST PetscSignalHandler_Private); */
#endif
    SignalSet = PETSC_TRUE;
  }
  if (!routine) {
#if !defined(PETSC_MISSING_SIGALRM)
    /* signal(SIGALRM, 0); */
#endif
#if !defined(PETSC_MISSING_SIGBUS)
    signal(SIGBUS,  0);
#endif
#if !defined(PETSC_MISSING_SIGCONT)
    /* signal(SIGCONT, 0); */
#endif
#if !defined(PETSC_MISSING_SIGFPE)
    signal(SIGFPE,  0);
#endif
#if !defined(PETSC_MISSING_SIGHUP)
    signal(SIGHUP,  0);
#endif
#if !defined(PETSC_MISSING_SIGILL)
    signal(SIGILL,  0);
#endif
#if !defined(PETSC_MISSING_SIGINT)
    /* signal(SIGINT,  0); */
#endif
#if !defined(PETSC_MISSING_SIGPIPE)
    signal(SIGPIPE, 0);
#endif
#if !defined(PETSC_MISSING_SIGQUIT)
    signal(SIGQUIT, 0);
#endif
#if !defined(PETSC_MISSING_SIGSEGV)
    signal(SIGSEGV, 0);
#endif
#if !defined(PETSC_MISSING_SIGSYS)
    signal(SIGSYS,  0);
#endif
#if !defined(PETSC_MISSING_SIGTERM)
    signal(SIGTERM, 0);
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
    signal(SIGTRAP, 0);
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
    /* signal(SIGTSTP, 0); */
#endif
#if !defined(PETSC_MISSING_SIGURG)
    signal(SIGURG,  0);
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
    /*    signal(SIGUSR1, 0); */
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
    /* signal(SIGUSR2, 0); */
#endif
    SignalSet = PETSC_FALSE;
  }
  ierr = PetscNew(struct SH,&newsh);CHKERRQ(ierr);
  if (sh) {
    if (sh->classid != SIGNAL_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Signal object has been corrupted");
    newsh->previous = sh;
  } 
  else {newsh->previous = 0;}
  newsh->handler = routine;
  newsh->ctx     = ctx;
  newsh->classid  = SIGNAL_CLASSID;
  sh             = newsh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscPopSignalHandler"
/*@
   PetscPopSignalHandler - Removes the most last signal handler that was pushed.
       If no signal handlers are left on the stack it will remove the PETSc signal handler.
       (That is PETSc will no longer catch signals).

   Not Collective

  Level: developer

   Concepts: signal handler^setting

.seealso: PetscPushSignalHandler()

@*/
PetscErrorCode  PetscPopSignalHandler(void)
{
  struct SH *tmp;

  PetscFunctionBegin;
  if (!sh) PetscFunctionReturn(0);
  if (sh->classid != SIGNAL_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Signal object has been corrupted");

  tmp = sh;
  sh  = sh->previous;
  PetscFreeVoid(tmp);
  if (!sh || !sh->handler) {
#if !defined(PETSC_MISSING_SIGALRM)
    /* signal(SIGALRM, 0); */
#endif
#if !defined(PETSC_MISSING_SIGBUS)
    signal(SIGBUS,  0);
#endif
#if !defined(PETSC_MISSING_SIGCONT)
    /* signal(SIGCONT, 0); */
#endif
#if !defined(PETSC_MISSING_SIGFPE)
    signal(SIGFPE,  0);
#endif
#if !defined(PETSC_MISSING_SIGHUP)
    signal(SIGHUP,  0);
#endif
#if !defined(PETSC_MISSING_SIGILL)
    signal(SIGILL,  0);
#endif
#if !defined(PETSC_MISSING_SIGINT)
    /* signal(SIGINT,  0); */
#endif
#if !defined(PETSC_MISSING_SIGPIPE)
    signal(SIGPIPE, 0);
#endif
#if !defined(PETSC_MISSING_SIGQUIT)
    signal(SIGQUIT, 0);
#endif
#if !defined(PETSC_MISSING_SIGSEGV)
    signal(SIGSEGV, 0);
#endif
#if !defined(PETSC_MISSING_SIGSYS)
    signal(SIGSYS,  0);
#endif
#if !defined(PETSC_MISSING_SIGTERM)
    signal(SIGTERM, 0);
#endif
#if !defined(PETSC_MISSING_SIGTRAP)
    signal(SIGTRAP, 0);
#endif
#if !defined(PETSC_MISSING_SIGTSTP)
    /* signal(SIGTSTP, 0); */
#endif
#if !defined(PETSC_MISSING_SIGURG)
    signal(SIGURG,  0);
#endif
#if !defined(PETSC_MISSING_SIGUSR1)
    /*    signal(SIGUSR1, 0); */
#endif
#if !defined(PETSC_MISSING_SIGUSR2)
    /* signal(SIGUSR2, 0); */
#endif
    SignalSet = PETSC_FALSE;
  } else {
    SignalSet = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}




