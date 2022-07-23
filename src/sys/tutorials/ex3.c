
static char help[] = "Augmenting PETSc profiling by add events.\n\
Run this program with one of the\n\
following options to generate logging information:  -log, -log_view,\n\
-log_all.  The PETSc routines automatically log event times and flops,\n\
so this monitoring is intended solely for users to employ in application\n\
codes.\n\n";

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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  /*
     Create a new user-defined event.
      - Note that PetscLogEventRegister() returns to the user a unique
        integer event number, which should then be used for profiling
        the event via PetscLogEventBegin() and PetscLogEventEnd().
      - The user can also optionally log floating point operations
        with the routine PetscLogFlops().
  */
  PetscCall(PetscLogEventRegister("User event",PETSC_VIEWER_CLASSID,&USER_EVENT));
  PetscCall(PetscLogEventGetId("User event",&check_USER_EVENT));
  PetscCheck(USER_EVENT == check_USER_EVENT,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Event Ids do not match");

  PetscCall(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  icount = 0;
  for (i=0; i<imax; i++) icount++;
  PetscCall(PetscLogFlops(imax));
  PetscCall(PetscSleep(0.5));
  PetscCall(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  /*
     We disable the logging of an event.

  */
  PetscCall(PetscLogEventDeactivate(USER_EVENT));
  PetscCall(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  PetscCall(PetscSleep(0.5));
  PetscCall(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  /*
     We next enable the logging of an event
  */
  PetscCall(PetscLogEventActivate(USER_EVENT));
  PetscCall(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  PetscCall(PetscSleep(0.5));
  PetscCall(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  /*
     We test event logging imbalance
  */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (rank == 0) PetscCall(PetscSleep(0.5));
  PetscCall(PetscLogEventSync(USER_EVENT,PETSC_COMM_WORLD));
  PetscCall(PetscLogEventBegin(USER_EVENT,0,0,0,0));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscSleep(0.5));
  PetscCall(PetscLogEventEnd(USER_EVENT,0,0,0,0));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: defined(PETSC_USE_LOG)

   test:

TEST*/
