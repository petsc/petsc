#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.17 1996/08/06 04:01:44 bsmith Exp curfman $";
#endif

static char help[] = "Demonstrates how users can augment the PETSc profiling by\n\
inserting their own event logging.  Run this program with one of the\n\
following options to generate logging information:  -log, -log_summary,\n\
-log_all.  The PETSc routines automatically log event times and flops,\n\
so this monitoring is intended solely for users to employ in application\n\
codes.  Note that the code must be compiled with the flag -DPETSC_LOG\n\
(the default) to activate logging.\n\n";

/*T
   Concepts: Plog (user-defined event logging)
   Concepts: Plog (activating/deactivating events)
   Routines: PLogEventRegister(); PLogEventBegin(); PLogEventEnd();
   Routines: PLogEventDeactivate(); PLogEventActivate(); PLogFlops();
   Routines: PLogEventMPEDeactivate(); PLogEventMPEActivate();
   Routines: PetscSleep();
   Processors: n
T*/

/* 
  Include "plog.h" so that we can use PETSc profiling routines.  Note 
  that this file automatically includes:
     petsc.h  - base PETSc routines
  Also, note that vec.h automatically includes plog.h, so generally
  plog.h need not be explicitly specified in the user's program.
*/
#include "plog.h"
#include <stdio.h>

int main(int argc,char **argv)
{
  int i, imax=10000, icount, USER_EVENT;

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* 
     Create a new user-defined event.
      - Note that PLogEventRegister() returns to the user a unique
        integer event number, which should then be used for profiling
        the event via PLogEventBegin() and PLogEventEnd().
      - The user can also optionally log floating point operations
        with the routine PLogFlops().
  */
  PLogEventRegister(&USER_EVENT,"User event      ","Red:");
  PLogEventBegin(USER_EVENT,0,0,0,0);
    icount = 0;
    for (i=0; i<imax; i++) icount++;
    PLogFlops(imax);
    PetscSleep(1);
  PLogEventEnd(USER_EVENT,0,0,0,0);

  /* 
     We disable the logging of an event.
      - Note that activation/deactivation of PETSc events and MPE 
        events is handled separately.
      - Note that the user can activate/deactive both user-defined
        events and predefined PETSc events.
  */
  PLogEventMPEDeactivate(USER_EVENT);
  PLogEventDeactivate(USER_EVENT);
  PLogEventBegin(USER_EVENT,0,0,0,0);
  PetscSleep(1);
  PLogEventEnd(USER_EVENT,0,0,0,0);

  /* 
     We next enable the logging of an event
  */
  PLogEventMPEActivate(USER_EVENT);
  PLogEventActivate(USER_EVENT);
  PLogEventBegin(USER_EVENT,0,0,0,0);
  PetscSleep(1);
  PLogEventEnd(USER_EVENT,0,0,0,0);

  PetscFinalize();
  return 0;
}
 
