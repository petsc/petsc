
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
  PetscMPIInt    rank;
  int            i,imax=10000,icount;
  PetscLogEvent  USER_EVENT,check_USER_EVENT;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  /*
     Create a new user-defined event.
      - Note that PetscLogEventRegister() returns to the user a unique
        integer event number, which should then be used for profiling
        the event via PetscLogEventBegin() and PetscLogEventEnd().
      - The user can also optionally log floating point operations
        with the routine PetscLogFlops().
  */
  CHKERRQ(PetscLogEventRegister("User event",PETSC_VIEWER_CLASSID,&USER_EVENT));
  CHKERRQ(PetscLogEventGetId("User event",&check_USER_EVENT));
  PetscCheckFalse(USER_EVENT != check_USER_EVENT,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Event Ids do not match");

  CHKERRQ(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  icount = 0;
  for (i=0; i<imax; i++) icount++;
  CHKERRQ(PetscLogFlops(imax));
  CHKERRQ(PetscSleep(0.5));
  CHKERRQ(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  /*
     We disable the logging of an event.

  */
  CHKERRQ(PetscLogEventDeactivate(USER_EVENT));
  CHKERRQ(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  CHKERRQ(PetscSleep(0.5));
  CHKERRQ(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  /*
     We next enable the logging of an event
  */
  CHKERRQ(PetscLogEventActivate(USER_EVENT));
  CHKERRQ(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  CHKERRQ(PetscSleep(0.5));
  CHKERRQ(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  /*
     We test event logging imbalance
  */
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) CHKERRQ(PetscSleep(0.5));
  CHKERRQ(PetscLogEventSync(USER_EVENT,PETSC_COMM_WORLD));
  CHKERRQ(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  CHKERRQ(PetscSleep(0.5));
  CHKERRQ(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: defined(PETSC_USE_LOG)

   test:

TEST*/
