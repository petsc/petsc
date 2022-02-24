
static char help[] = "Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()\n";

#include <petscsys.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1));
  CHKERRQ(PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
