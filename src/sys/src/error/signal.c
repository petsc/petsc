
/*
      Routines to handle signals the program will receive. 
    Usually this will call the error handlers.
*/
#include "sys.h"
#include <signal.h>          /*I <signal.h> I*/

static char *SIGNAME[] = { "Unknown", "HUP",  "INT",  "QUIT", "ILL",
                           "TRAP",    "ABRT", "EMT",  "FPE",  "KILL", 
                           "BUS error",  
                           "SEGV segmentation violation", 
                           "SYS",  "PIPE", "ALRM",
                           "TERM",    "URG",  "STOP", "TSTP", "CONT", 
                           "CHLD" }; 


/*
  PetscDefaultSignalHandler - Default signal handler.

  Input Parameters:
. sig   - signal value
. code,scp,addr - see the signal man page
*/
void PetscDefaultSignalHandler( int sig, int code,struct sigcontext * scp, 
                                char *addr )
{
  static char buf[128];

  signal( sig, SIG_DFL );
  if (sig >= 0 && sig <= 20) 
    sprintf( buf, "Error: Caught signal %s", SIGNAME[sig] );
  else
    strcpy( buf, "Error: Caught signal " );
  SETERR(1,buf);
}

/*@
   PetscSetDefaultSignals - Set up to catch the usual fatal errors and 
   kill the job..

   Input parameter:
.  routine - routine to call when a signal is received.  This should
             have a form that is compatible with "signal".
@*/
int PetscSetDefaultSignals( 
                       void (*routine)(int, int, struct sigcontext *,char*) )
{
  if (routine == 0) routine = PetscDefaultSignalHandler;
  signal( SIGQUIT, routine );
  signal( SIGILL,  routine );
  signal( SIGFPE,  routine );
  signal( SIGBUS,  routine );
  signal( SIGSEGV, routine );
  signal( SIGSYS,  routine );
  return 0;
}


