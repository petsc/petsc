/*$Id: ex3.c,v 1.32 2000/09/06 22:19:06 balay Exp balay $*/

static char help[] = "Demonstrates how users can augment the PETSc profiling by\n\
inserting their own event logging.  Run this program with one of the\n\
following options to generate logging information:  -log, -log_summary,\n\
-log_all.  The PETSc routines automatically log event times and flops,\n\
so this monitoring is intended solely for users to employ in application\n\
codes.  Note that the code must be compiled with the flag -DPETSC_USE_LOG\n\
(the default) to activate logging.\n\n";

/*T
   Concepts: PLog^User-defined event profiling (basic example);
   Concepts: PLog^Activating/deactivating events for profiling (basic example);
   Routines: PLogEventRegister(); PLogEventBegin(); PLogEventEnd();
   Routines: PLogEventDeactivate(); PLogEventActivate(); PLogFlops();
   Routines: PLogEventMPEDeactivate(); PLogEventMPEActivate();
   Routines: PetscSleep();
   Processors: n
T*/

/* 
  Include "petsc.h" so that we can use PETSc profiling routines.
*/
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int i,ierr,imax=10000,icount,USER_EVENT;

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* 
     Create a new user-defined event.
      - Note that PLogEventRegister() returns to the user a unique
        integer event number, which should then be used for profiling
        the event via PLogEventBegin() and PLogEventEnd().
      - The user can also optionally log floating point operations
        with the routine PLogFlops().
  */
  ierr = PLogEventRegister(&USER_EVENT,"User event","Red:");CHKERRA(ierr);
  ierr = PLogEventBegin(USER_EVENT,0,0,0,0);CHKERRA(ierr);
  icount = 0;
  for (i=0; i<imax; i++) icount++;
  ierr = PLogFlops(imax);CHKERRA(ierr);
  ierr = PetscSleep(1);CHKERRA(ierr);
  ierr = PLogEventEnd(USER_EVENT,0,0,0,0);CHKERRA(ierr);

  /* 
     We disable the logging of an event.
      - Note that activation/deactivation of PETSc events and MPE 
        events is handled separately.
      - Note that the user can activate/deactive both user-defined
        events and predefined PETSc events.
  */
  ierr = PLogEventMPEDeactivate(USER_EVENT);CHKERRA(ierr);
  ierr = PLogEventDeactivate(USER_EVENT);CHKERRA(ierr);
  ierr = PLogEventBegin(USER_EVENT,0,0,0,0);CHKERRA(ierr);
  ierr = PetscSleep(1);CHKERRA(ierr);
  ierr = PLogEventEnd(USER_EVENT,0,0,0,0);CHKERRA(ierr);

  /* 
     We next enable the logging of an event
  */
  ierr = PLogEventMPEActivate(USER_EVENT);CHKERRA(ierr);
  ierr = PLogEventActivate(USER_EVENT);CHKERRA(ierr);
  ierr = PLogEventBegin(USER_EVENT,0,0,0,0);CHKERRA(ierr);
  ierr = PetscSleep(1);CHKERRA(ierr);
  ierr = PLogEventEnd(USER_EVENT,0,0,0,0);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
