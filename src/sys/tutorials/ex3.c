
static char help[] = "Augmenting PETSc profiling by add events.\n\
Run this program with one of the\n\
following options to generate logging information:  -log, -log_view,\n\
-log_all.  The PETSc routines automatically log event times and flops,\n\
so this monitoring is intended solely for users to employ in application\n\
codes.\n\n";

/*T
   Concepts: PetscLog^user-defined event profiling
   Concepts: profiling^user-defined event
   Concepts: PetscLog^activating/deactivating events for profiling
   Concepts: profiling^activating/deactivating events
   Processors: n
T*/

/*
  Include "petscsys.h" so that we can use PETSc profiling routines.
*/
#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  int            i,imax=10000,icount;
  PetscLogEvent  USER_EVENT,check_USER_EVENT;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  /*
     Create a new user-defined event.
      - Note that PetscLogEventRegister() returns to the user a unique
        integer event number, which should then be used for profiling
        the event via PetscLogEventBegin() and PetscLogEventEnd().
      - The user can also optionally log floating point operations
        with the routine PetscLogFlops().
  */
  ierr = PetscLogEventRegister("User event",PETSC_VIEWER_CLASSID,&USER_EVENT);CHKERRQ(ierr);
  ierr = PetscLogEventGetId("User event",&check_USER_EVENT);CHKERRQ(ierr);
  if (USER_EVENT != check_USER_EVENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Event Ids do not match");

  ierr = PetscLogEventBegin(USER_EVENT,0,0,0,0);CHKERRQ(ierr);
  icount = 0;
  for (i=0; i<imax; i++) icount++;
  ierr = PetscLogFlops(imax);CHKERRQ(ierr);
  ierr = PetscSleep(0.5);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(USER_EVENT,0,0,0,0);CHKERRQ(ierr);

  /*
     We disable the logging of an event.

  */
  ierr = PetscLogEventDeactivate(USER_EVENT);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(USER_EVENT,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscSleep(0.5);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(USER_EVENT,0,0,0,0);CHKERRQ(ierr);

  /*
     We next enable the logging of an event
  */
  ierr = PetscLogEventActivate(USER_EVENT);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(USER_EVENT,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscSleep(0.5);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(USER_EVENT,0,0,0,0);CHKERRQ(ierr);

  /*
     We test event logging imbalance
  */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (!rank) {ierr = PetscSleep(0.5);CHKERRQ(ierr);}
  ierr = PetscLogEventSync(USER_EVENT,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(USER_EVENT,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = PetscSleep(0.5);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(USER_EVENT,0,0,0,0);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: define(PETSC_USE_LOG)

   test:

TEST*/
