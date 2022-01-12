
static char help[] = "Tests PetscCommGetComm().\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  MPI_Comm       comms[10],comm;
  PetscInt       i;
  PetscRandom    rand;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)rand,&comm);CHKERRQ(ierr);
  for (i=0; i<10; i++) {
    ierr = PetscCommGetComm(comm,&comms[i]);CHKERRQ(ierr);
  }
  for (i=0; i<5; i++) {
    ierr = PetscCommRestoreComm(comm,&comms[i]);CHKERRQ(ierr);
  }
  for (i=0; i<5; i++) {
    ierr = PetscCommGetComm(comm,&comms[i]);CHKERRQ(ierr);
  }
  for (i=0; i<10; i++) {
    ierr = PetscCommRestoreComm(comm,&comms[i]);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     requires: defined(PETSC_USE_LOG)
     args: -info
     filter: grep Reusing | wc -l

TEST*/
