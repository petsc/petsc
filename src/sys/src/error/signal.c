#ifndef lint
static char vcid[] = "$Id: signal.c,v 1.12 1995/07/17 20:40:20 bsmith Exp curfman $";
#endif
/*
      Routines to handle signals the program will receive. 
    Usually this will call the error handlers.
*/
#include "petsc.h"
#include "sys.h"
#include <signal.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "petscfix.h"     

struct SH {
  int    cookie;
  int    (*handler)(int,void *);
  void   *ctx;
  struct SH* previous;
};
static struct SH* sh        = 0;
static int        SignalSet = 0;

static char *SIGNAME[] = { "Unknown", "HUP",  "INT",  "QUIT", "ILL",
                           "TRAP",    "ABRT", "EMT",  "FPE",  "KILL", 
                           "BUS error",  
                           "SEGV segmentation violation", 
                           "SYS",  "PIPE", "ALRM",
                           "TERM",    "URG",  "STOP", "TSTP", "CONT", 
                           "CHLD" }; 

/*
    This is the signal handler called by the system. This calls 
  your signal handler.
*/
#if defined(PARCH_IRIX) || defined(PARCH_sun4)
static void PetscSignalHandler( int sig, int code,
                                        struct sigcontext * scp,char *addr)
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
  if (ierr) MPI_Abort(MPI_COMM_WORLD,0);
}


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
    sprintf( buf, "Error: Caught signal %s\n", SIGNAME[sig] );
  else
    strcpy( buf, "Error: Caught signal\n" );
  strcat(buf,"Try option -start_in_debugger or -on_error_attach_debugger");
  strcat(buf," to determine where problem occurs");
  ierr =  PetscError(0,0,"Unknown",buf,1);
  MPI_Abort(MPI_COMM_WORLD,ierr);
  return 0;
}

/*@
   PetscPushSignalHandler - Set up to catch the usual fatal errors and 
   kill the job.

   Input Parameter:
.  routine - routine to call when a signal is received
.  ctx - optional context needed by the routine

.keywords: push, signal, handler
@*/
int PetscPushSignalHandler(int (*routine)(int, void*),void* ctx )
{
  struct  SH *newsh;
  if (!SignalSet && routine) {
#if defined(PARCH_IRIX) && defined(__cplusplus)
    signal( SIGQUIT, (void (*)(...)) PetscSignalHandler );
    signal( SIGILL,  (void (*)(...)) PetscSignalHandler );
    signal( SIGFPE,  (void (*)(...)) PetscSignalHandler );
    signal( SIGSEGV, (void (*)(...)) PetscSignalHandler );
    signal( SIGSYS,  (void (*)(...)) PetscSignalHandler );
#else
    signal( SIGQUIT, PetscSignalHandler );
    signal( SIGILL,  PetscSignalHandler );
    signal( SIGFPE,  PetscSignalHandler );
    signal( SIGBUS,  PetscSignalHandler );
    signal( SIGSEGV, PetscSignalHandler );
    signal( SIGSYS,  PetscSignalHandler );
#endif
    SignalSet = 1;
  }
  if (!routine) {
    signal( SIGQUIT, 0 );
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
#if !defined(PARCH_IRIX) && !defined(__cplusplus)
    signal( SIGBUS,  0 );
#endif
    signal( SIGSEGV, 0 );
    signal( SIGSYS,  0 );
    SignalSet = 0;
  }
  newsh = (struct SH*) PETSCMALLOC(sizeof(struct SH)); CHKPTRQ(newsh);
  if (sh) {newsh->previous = sh;} 
  else {newsh->previous = 0;}
  newsh->handler = routine;
  newsh->ctx     = ctx;
  sh = newsh;
  return 0;
}

int PetscPopSignalHandler()
{
  struct SH *tmp;
  if (!sh) return 0;
  tmp = sh;
  sh  = sh->previous;
  PETSCFREE(tmp);
  if (!sh || !sh->handler) {
    signal( SIGQUIT, 0 );
    signal( SIGILL,  0 );
    signal( SIGFPE,  0 );
#if !defined(PARCH_IRIX) && !defined(__cplusplus)
    signal( SIGBUS,  0 );
#endif
    signal( SIGSEGV, 0 );
    signal( SIGSYS,  0 );
    SignalSet = 0;
  }
  else {
    SignalSet = 1;
  }
  return 0;
}
