
static char help[] = "Tests the different MatColoring implementations and ISColoringTestValid() \n\
                      Modified from the code contributed by Ali Berk Kahraman. \n\n";
#include <petscmat.h>

PetscErrorCode FormJacobian(Mat A)
{
  PetscInt    M, ownbegin, ownend, i, j;
  PetscScalar dummy = 0.0;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(A, &M, NULL));
  PetscCall(MatGetOwnershipRange(A, &ownbegin, &ownend));

  for (i = ownbegin; i < ownend; i++) {
    for (j = i - 3; j < i + 3; j++) {
      if (j >= 0 && j < M) PetscCall(MatSetValues(A, 1, &i, 1, &j, &dummy, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  Mat         J;
  PetscMPIInt size;
  PetscInt    M = 8;
  ISColoring  iscoloring;
  MatColoring coloring;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, M, M));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(FormJacobian(J));
  PetscCall(MatView(J, PETSC_VIEWER_STDOUT_WORLD));

  /*
    Color the matrix, i.e. determine groups of columns that share no common
    rows. These columns in the Jacobian can all be computed simultaneously.
   */
  PetscCall(MatColoringCreate(J, &coloring));
  PetscCall(MatColoringSetType(coloring, MATCOLORINGGREEDY));
  PetscCall(MatColoringSetFromOptions(coloring));
  PetscCall(MatColoringApply(coloring, &iscoloring));

  if (size == 1) PetscCall(MatISColoringTest(J, iscoloring));

  PetscCall(ISColoringDestroy(&iscoloring));
  PetscCall(MatColoringDestroy(&coloring));
  PetscCall(MatDestroy(&J));
  PetscCall(PetscFinalize());
  return 0;
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
