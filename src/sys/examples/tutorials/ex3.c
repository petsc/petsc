/*$Id: ex3.c,v 1.35 2000/09/27 03:30:36 bsmith Exp bsmith $*/

static char help[] = "Augmenting PETSc profiling by add events.\n\
Run this program with one of the\n\
following options to generate logging information:  -log, -log_summary,\n\
-log_all.  The PETSc routines automatically log event times and flops,\n\
so this monitoring is intended solely for users to employ in application\n\
codes.  Note that the code must be compiled with the flag -DPETSC_USE_LOG\n\
(the default) to activate logging.\n\n";

/*T
   Concepts: PetscLog^user-defined event profiling
   Concepts: profiling^user-defined event
   Concepts: PetscLog^activating/deactivating events for profiling
   Concepts: profiling^activating/deactivating events
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
      - Note that PetscLogEventRegister() returns to the user a unique
        integer event number, which should then be used for profiling
        the event via PetscLogEventBegin() and PetscLogEventEnd().
      - The user can also optionally log floating point operations
        with the routine PetscLogFlops().
  */
  ierr = PetscLogEventRegister(&USER_EVENT,"User event","Red:");CHKERRA(ierr);
  ierr = PetscLogEventBegin(USER_EVENT,0,0,0,0);CHKERRA(ierr);
  icount = 0;
  for (i=0; i<imax; i++) icount++;
  ierr = PetscLogFlops(imax);CHKERRA(ierr);
  ierr = PetscSleep(1);CHKERRA(ierr);
  ierr = PetscLogEventEnd(USER_EVENT,0,0,0,0);CHKERRA(ierr);

  /* 
     We disable the logging of an event.
      - Note that activation/deactivation of PETSc events and MPE 
        events is handled separately.
      - Note that the user can activate/deactive both user-defined
        events and predefined PETSc events.
  */
  ierr = PetscLogEventMPEDeactivate(USER_EVENT);CHKERRA(ierr);
  ierr = PetscLogEventDeactivate(USER_EVENT);CHKERRA(ierr);
  ierr = PetscLogEventBegin(USER_EVENT,0,0,0,0);CHKERRA(ierr);
  ierr = PetscSleep(1);CHKERRA(ierr);
  ierr = PetscLogEventEnd(USER_EVENT,0,0,0,0);CHKERRA(ierr);

  /* 
     We next enable the logging of an event
  */
  ierr = PetscLogEventMPEActivate(USER_EVENT);CHKERRA(ierr);
  ierr = PetscLogEventActivate(USER_EVENT);CHKERRA(ierr);
  ierr = PetscLogEventBegin(USER_EVENT,0,0,0,0);CHKERRA(ierr);
  ierr = PetscSleep(1);CHKERRA(ierr);
  ierr = PetscLogEventEnd(USER_EVENT,0,0,0,0);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
