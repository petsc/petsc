#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: signal.c,v 1.46 1997/09/18 18:11:55 balay Exp bsmith $";
#endif
/*
      Routines to handle signals the program will receive. 
    Usually this will call the error handlers.
*/
#include <signal.h>
#include "petsc.h"             /*I   "petsc.h"   I*/
#include "sys.h"
#include "pinclude/petscfix.h"     

struct SH {
  int    cookie;
  int    (*handler)(int,void *);
  void   *ctx;
  struct SH* previous;
};
static struct SH* sh        = 0;
static int        SignalSet = 0;

static char *SIGNAME[] = { "unknown", "HUP",  "INT",  "QUIT", "ILL",
                           "TRAP",    "ABRT", "EMT",  
                           "FPE:\nPETSC ERROR: Floating Point Exception, probably divide by zero",
                           "KILL", 
                           "BUS: Bus Error",  
                           "SEGV:\nPETSC ERROR: Segmentation Violation, probably memory corruption", 
                           "SYS",  "PIPE", "ALRM",
                           "TERM",    "URG",  "STOP", "TSTP", "CONT", 
                           "CHLD" }; 

#undef __FUNC__  
#define __FUNC__ "PetscSignalHandler"
/*
    This is the signal handler called by the system. This calls 
  your signal handler.
*/
#if defined(PARCH_IRIX)  || defined(PARCH_IRIX64) || defined(PARCH_IRIX5)|| defined(PARCH_sun4)
static void PetscSignalHandler( int sig, int code,struct sigcontext * scp,char *addr)
#else
static void PetscSignalHandler( int sig )
#endif
{
  int ierr;
  if (!sh || !sh->handler) {
    ierr = PetscDefaultSignalHandler(sig,(void*)0);
  }
  else{
    ierr = (*sh->handler)(sig,sh->ctx);
  }
  if (ierr) MPI_Abort(PETSC_COMM_WORLD,0);
}


#undef __FUNC__  
#define __FUNC__ "PetscDefaultSignalHandler"
/*@
   PetscDefaultSignalHandler - Default signal handler.

   Input Parameters:
.  sig - signal value
.  ptr - unused pointer

.keywords: default, signal, handler
@*/
int PetscDefaultSignalHandler( int sig, void *ptr)
{
  int ierr;
  static char buf[128];
  signal( sig, SIG_DFL );
  if (sig >= 0 && sig <= 20) 
    sprintf( buf, "Caught signal %s\n", SIGNAME[sig] );
  else
    PetscStrcpy( buf, "Caught signal\n" );
  PetscStrcat(buf,"PETSC ERROR: Try option -start_in_debugger or ");
  PetscStrcat(buf,"-on_error_attach_debugger ");
  PetscStrcat(buf,"to\nPETSC ERROR: determine where problem occurs");
  ierr =  PetscError(0,"unknownfunction","unknown file"," ",PETSC_ERR_SIG,0,buf);
  MPI_Abort(PETSC_COMM_WORLD,ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscPushSignalHandler"
/*@C
   PetscPushSignalHandler - Catches the usual fatal errors and 
   calls a user-provided routine.

   Input Parameter:
.  routine - routine to call when a signal is received
.  ctx - optional context needed by the routine

.keywords: push, signal, handler
@*/
int PetscPushSignalHandler(int (*routine)(int, void*),void* ctx )
{
  struct  SH *newsh;
  if (!SignalSet && routine) {
#if defined(PARCH_IRIX5)  && defined(__cplusplus)
    signal( SIGQUIT, (void (*)(...)) PetscSignalHandler );
    signal( SIGILL,  (void (*)(...)) PetscSignalHandler );
    signal( SIGFPE,  (void (*)(...)) PetscSignalHandler );
    signal( SIGSEGV, (void (*)(...)) PetscSignalHandler );
    signal( SIGSYS,  (void (*)(...)) PetscSignalHandler );
#elif (defined(PARCH_IRIX64) || defined(PARCH_IRIX)) && defined(__cplusplus)
    signal( SIGQUIT, (void (*)(int)) PetscSignalHandler );
    signal( SIGILL,  (void (*)(int)) PetscSignalHandler );
    signal( SIGFPE,  (void (*)(int)) PetscSignalHandler );
    signal( SIGSEGV, (void (*)(int)) PetscSignalHandler );
    signal( SIGSYS,  (void (*)(int)) PetscSignalHandler );
#elif defined(PARCH_nt)
    /*
    signal( SIGILL,  PetscSignalHandler );
    signal( SIGFPE,  PetscSignalHandler );
    signal( SIGSEGV, PetscSignalHandler );
    */
#elif defined(PARCH_nt_gnu) || defined (PARCH_linux) 
    signal( SIGILL,  PetscSignalHandler );
    signal( SIGFPE,  PetscSignalHandler );
    signal( SIGSEGV, PetscSignalHandler );
    signal( SIGBUS,  PetscSignalHandler );
    signal( SIGQUIT, PetscSignalHandler );
#else
    signal( SIGILL,  PetscSignalHandler );
    signal( SIGFPE,  PetscSignalHandler );
    signal( SIGSEGV, PetscSignalHandler );
    signal( SIGBUS,  PetscSignalHandler );
    signal( SIGQUIT, PetscSignalHandler );
    signal( SIGSYS,  PetscSignalHandler );
#endif
    SignalSet = 1;
  }
  if (!routine) {
#if (defined(PARCH_IRIX)  || defined(PARCH_IRIX5) || defined(PARCH_IRIX64)) && defined(__cplusplus)
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGQUIT, 0 );
    signal( SIGSEGV, 0 );
    signal( SIGSYS,  0 );
#elif defined(PARCH_nt)
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGSEGV, 0 );
#elif defined(PARCH_nt_gnu) || defined (PARCH_linux) 
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGQUIT, 0 );
    signal( SIGSEGV, 0 );
    signal( SIGBUS,  0 );
#else
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGQUIT, 0 );
    signal( SIGSEGV, 0 );
    signal( SIGBUS,  0 );
    signal( SIGSYS,  0 );
#endif
    SignalSet = 0;
  }
  newsh = (struct SH*) PetscMalloc(sizeof(struct SH)); CHKPTRQ(newsh);
  if (sh) {newsh->previous = sh;} 
  else {newsh->previous = 0;}
  newsh->handler = routine;
  newsh->ctx     = ctx;
  sh = newsh;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscPopSignalHandler"
int PetscPopSignalHandler()
{
  struct SH *tmp;
  if (!sh) return 0;
  tmp = sh;
  sh  = sh->previous;
  PetscFree(tmp);
  if (!sh || !sh->handler) {
#if (defined(PARCH_IRIX)  || defined(PARCH_IRIX5) || defined(PARCH_IRIX64)) && defined(__cplusplus)
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGQUIT, 0 );
    signal( SIGSEGV, 0 );
    signal( SIGSYS,  0 );
#elif defined(PARCH_nt)
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGSEGV, 0 );
#elif defined(PARCH_nt_gnu) || defined (PARCH_linux) 
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGQUIT, 0 );
    signal( SIGSEGV, 0 );
    signal( SIGBUS,  0 );
#else
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
    signal( SIGQUIT, 0 );
    signal( SIGSEGV, 0 );
    signal( SIGBUS,  0 );
    signal( SIGSYS,  0 );
#endif
    SignalSet = 0;
  }
  else {
    SignalSet = 1;
  }
  return 0;
}

