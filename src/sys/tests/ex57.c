
static char help[] = "Tests PetscCommGetComm().\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  MPI_Comm       comms[10],comm;
  PetscInt       i;
  PetscRandom    rand;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
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
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     requires: defined(PETSC_USE_LOG)
     args: -info
     filter: grep Reusing | wc -l

TEST*/
