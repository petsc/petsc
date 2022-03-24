
static char help[] = "Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()\n";

#include <petscsys.h>

int main(int argc,char **args)
{

  CHKERRQ(PetscInitialize(&argc,&args,NULL,help));
  CHKERRQ(PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1));
  CHKERRQ(PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
