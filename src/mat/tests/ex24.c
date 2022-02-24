
static char help[] = "Tests the different MatColoring implementatons and ISColoringTestValid() \n\
                      Modifed from the code contributed by Ali Berk Kahraman. \n\n";
#include <petscmat.h>

PetscErrorCode FormJacobian(Mat A)
{
  PetscInt       M,ownbegin,ownend,i,j;
  PetscScalar    dummy=0.0;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(A,&M,NULL));
  CHKERRQ(MatGetOwnershipRange(A,&ownbegin,&ownend));

  for (i=ownbegin; i<ownend; i++) {
    for (j=i-3; j<i+3; j++) {
      if (j >= 0 && j < M) {
        CHKERRQ(MatSetValues(A,1,&i,1,&j,&dummy,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  Mat            J;
  PetscMPIInt    size;
  PetscInt       M=8;
  ISColoring     iscoloring;
  MatColoring    coloring;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&J));
  CHKERRQ(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, M, M));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(MatSetUp(J));

  CHKERRQ(FormJacobian(J));
  CHKERRQ(MatView(J,PETSC_VIEWER_STDOUT_WORLD));

  /*
    Color the matrix, i.e. determine groups of columns that share no common
    rows. These columns in the Jacobian can all be computed simultaneously.
   */
  CHKERRQ(MatColoringCreate(J, &coloring));
  CHKERRQ(MatColoringSetType(coloring,MATCOLORINGGREEDY));
  CHKERRQ(MatColoringSetFromOptions(coloring));
  CHKERRQ(MatColoringApply(coloring, &iscoloring));

  if (size == 1) {
    CHKERRQ(MatISColoringTest(J,iscoloring));
  }

  CHKERRQ(ISColoringDestroy(&iscoloring));
  CHKERRQ(MatColoringDestroy(&coloring));
  CHKERRQ(MatDestroy(&J));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: sl
      requires: !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -mat_coloring_type sl
      output_file: output/ex24_1.out

   test:
      suffix: lf
      requires: !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -mat_coloring_type lf
      output_file: output/ex24_1.out

   test:
      suffix: id
      requires: !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -mat_coloring_type id
      output_file: output/ex24_1.out

TEST*/
