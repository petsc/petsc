
static char help[] = "Tests PetscCommGetComm().\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  MPI_Comm       comms[10],comm;
  PetscInt       i;
  PetscRandom    rand;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscObjectGetComm((PetscObject)rand,&comm));
  for (i=0; i<10; i++) {
    CHKERRQ(PetscCommGetComm(comm,&comms[i]));
  }
  for (i=0; i<5; i++) {
    CHKERRQ(PetscCommRestoreComm(comm,&comms[i]));
  }
  for (i=0; i<5; i++) {
    CHKERRQ(PetscCommGetComm(comm,&comms[i]));
  }
  for (i=0; i<10; i++) {
    CHKERRQ(PetscCommRestoreComm(comm,&comms[i]));
  }
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: defined(PETSC_USE_LOG)
     args: -info
     filter: grep Reusing | wc -l

TEST*/
