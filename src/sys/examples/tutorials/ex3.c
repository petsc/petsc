#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.15 1996/03/19 21:24:49 bsmith Exp bsmith $";
#endif

static char help[] = 
"Demonstrates how users can augment the PETSc profiling by\n\
inserting their own event logging.  Run this program with one of the\n\
following options to generate logging information:  -log, -log_summary,\n\
-log_all.  The PETSc routines automatically log event times and flops,\n\
so this monitoring is intended solely for users to employ in application\n\
codes.  Note that the code must be compiled with the flag -DPETSC_LOG\n\
(the default) to activate logging.\n\n";

#include "petsc.h"
#include "vec.h"
#include <stdio.h>

int main(int argc,char **argv)
{
  int i, imax=10000, icount;
  int USER_EVENT;

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* demonstrate creating a new user event */
  PLogEventRegister(&USER_EVENT,"User event      ","Red:");
  PLogEventBegin(USER_EVENT,0,0,0,0);
    icount = 0;
    for (i=0; i<imax; i++) icount++;
    PLogFlops(imax);
    PetscSleep(1);
  PLogEventEnd(USER_EVENT,0,0,0,0);

  /* demonstrate disabling the logging of an event */
  /* this is done for both MPE logging and PETSc logging. */
  PLogEventMPEDeActivate(USER_EVENT);
  PLogEventDeActivate(USER_EVENT);
  PLogEventBegin(USER_EVENT,0,0,0,0);
  PetscSleep(1);
  PLogEventEnd(USER_EVENT,0,0,0,0);

  /* demonstrate enabling the logging of an event */
  PLogEventMPEActivate(USER_EVENT);
  PLogEventActivate(USER_EVENT);
  PLogEventBegin(USER_EVENT,0,0,0,0);
  PetscSleep(1);
  PLogEventEnd(USER_EVENT,0,0,0,0);

  PetscFinalize();
  return 0;
}
 
