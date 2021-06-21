
static char help[] = "Tests PetscSynchronizedPrintf() and PetscSynchronizedFPrintf().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Greetings from %d\n",rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,PETSC_STDOUT,"Greetings again from %d\n",rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3

TEST*/
