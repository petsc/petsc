
/*
      Routines to handle signals the program will receive. 
    Usually this will call the error handlers.
*/
#include "petsc.h"
#include "sys.h"
#include <signal.h>          /*I <signal.h> I*/

static char *SIGNAME[] = { "Unknown", "HUP",  "INT",  "QUIT", "ILL",
                           "TRAP",    "ABRT", "EMT",  "FPE",  "KILL", 
                           "BUS error",  
                           "SEGV segmentation violation", 
                           "SYS",  "PIPE", "ALRM",
                           "TERM",    "URG",  "STOP", "TSTP", "CONT", 
                           "CHLD" }; 

#if !defined(PARCH_rs6000) && !defined(PARCH_freebsd) && !defined(PARCH_alpha)
/*
  PetscDefaultSignalHandler - Default signal handler.

  Input Parameters:
. sig   - signal value
. code,scp,addr - see the signal man page
*/
void PetscDefaultSignalHandler( int sig, int code,struct sigcontext * scp, 
                                char *addr )
{
#else
void PetscDefaultSignalHandler( int sig )
{
#endif
  int ierr;
  static char buf[128];

  signal( sig, SIG_DFL );
  if (sig >= 0 && sig <= 20) 
    sprintf( buf, "Error: Caught signal %s", SIGNAME[sig] );
  else
    strcpy( buf, "Error: Caught signal " );
  ierr = PetscError(0,"Unknown",buf,1);
  if (ierr) exit(ierr); else return;
}

#if !defined(PARCH_rs6000) && !defined(PARCH_freebsd) && !defined(PARCH_alpha)
/*@
   PetscSetSignalHandler - Set up to catch the usual fatal errors and 
   kill the job..

   Input parameter:
.  routine - routine to call when a signal is received.  This should
             have a form that is compatible with "signal". Not that 
             on the IBM rs6000 it takes one argument instead of the 
             traditional Unix four.
@*/
int PetscSetSignalHandler( 
                       void (*routine)(int, int, struct sigcontext *,char*) )
{
#else
int PetscSetSignalHandler( 
                       void (*routine)(int) )
{
#endif
  if (routine == 0) routine = PetscDefaultSignalHandler;
  signal( SIGQUIT, routine );
  signal( SIGILL,  routine );
  signal( SIGFPE,  routine );
  signal( SIGBUS,  routine );
  signal( SIGSEGV, routine );
  signal( SIGSYS,  routine );
  return 0;
}


