
static char help[] = "Tests PetscCommGetComm().\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  MPI_Comm       comms[10],comm;
  PetscInt       i;
  PetscRandom    rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  PetscCall(PetscObjectGetComm((PetscObject)rand,&comm));
  for (i=0; i<10; i++) {
    PetscCall(PetscCommGetComm(comm,&comms[i]));
  }
  for (i=0; i<5; i++) {
    PetscCall(PetscCommRestoreComm(comm,&comms[i]));
  }
  for (i=0; i<5; i++) {
    PetscCall(PetscCommGetComm(comm,&comms[i]));
  }
  for (i=0; i<10; i++) {
    PetscCall(PetscCommRestoreComm(comm,&comms[i]));
  }
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: defined(PETSC_USE_LOG)
     args: -info
     filter: grep Reusing | wc -l

TEST*/
