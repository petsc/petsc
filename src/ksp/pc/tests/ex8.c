
static char help[] = "Tests PCView() before PCSetup() with -pc_type lu.\n\n";

#include <petscmat.h>
#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            A;
  PC             pc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,1,1,1,1));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValue(A,0,0,1,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PCCreate(PETSC_COMM_WORLD,&pc));
  PetscCall(PCSetOperators(pc,A,A));
  PetscCall(PCSetType(pc,PCLU));
  PetscCall(PCView(pc,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
