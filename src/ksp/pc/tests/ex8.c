
static char help[] = "Tests PCView() before PCSetup() with -pc_type lu.\n\n";

#include <petscmat.h>
#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            A;
  PC             pc;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,1,1,1,1));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetValue(A,0,0,1,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PCCreate(PETSC_COMM_WORLD,&pc));
  CHKERRQ(PCSetOperators(pc,A,A));
  CHKERRQ(PCSetType(pc,PCLU));
  CHKERRQ(PCView(pc,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PCDestroy(&pc));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
