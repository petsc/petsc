
static char help[] = "Tests MatILUFactorSymbolic() on matrix with missing diagonal.\n\n";

#include <petscmat.h>
#include <petscpc.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j;
  PetscErrorCode ierr;
  PetscScalar    v;
  PC             pc;
  Vec            xtmp;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,3,3));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,3,&xtmp));
  i    = 0; j = 0; v = 4;
  CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 0; j = 2; v = 1;
  CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 1; j = 0; v = 1;
  CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 1; j = 1; v = 4;
  CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 2; j = 1; v = 1;
  CHKERRQ(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PCCreate(PETSC_COMM_WORLD,&pc));
  CHKERRQ(PCSetFromOptions(pc));
  CHKERRQ(PCSetOperators(pc,C,C));
  CHKERRQ(PCSetUp(pc));
  CHKERRQ(PCFactorGetMatrix(pc,&A));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PCDestroy(&pc));
  CHKERRQ(VecDestroy(&xtmp));
  CHKERRQ(MatDestroy(&C));

  ierr = PetscFinalize();
  return ierr;
}
