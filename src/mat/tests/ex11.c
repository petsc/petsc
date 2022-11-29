
static char help[] = "Tests the use of MatZeroRows() for uniprocessor matrices.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         C;
  PetscInt    i, j, m = 5, n = 5, Ii, J;
  PetscScalar v, five = 5.0;
  IS          isrow;
  PetscBool   keepnonzeropattern;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  /* create the matrix for the five point stencil, YET AGAIN*/
  PetscCall(MatCreate(PETSC_COMM_SELF, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      v  = -1.0;
      Ii = j + n * i;
      if (i > 0) {
        J = Ii - n;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      if (i < m - 1) {
        J = Ii + n;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      if (j > 0) {
        J = Ii - 1;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      if (j < n - 1) {
        J = Ii + 1;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, INSERT_VALUES));
      }
      v = 4.0;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &Ii, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(ISCreateStride(PETSC_COMM_SELF, (m * n) / 2, 0, 2, &isrow));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-keep_nonzero_pattern", &keepnonzeropattern));
  if (keepnonzeropattern) PetscCall(MatSetOption(C, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));

  PetscCall(MatZeroRowsIS(C, isrow, five, 0, 0));

  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(ISDestroy(&isrow));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -mat_type seqbaij -mat_block_size 5

   test:
      suffix: 3
      args: -keep_nonzero_pattern

   test:
      suffix: 4
      args: -keep_nonzero_pattern -mat_type seqbaij -mat_block_size 5

TEST*/
