
static char help[] = "Tests PetscSequentialPhaseBegin() and PetscSequentialPhaseEnd()\n";

#include <petscsys.h>

int main(int argc,char **args)
{

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCall(PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1));
  PetscCall(PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
