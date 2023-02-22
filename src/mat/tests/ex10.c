
static char help[] = "Tests repeated use of assembly for matrices.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         C, B;
  PetscInt    i, j, m = 5, n = 2, Ii, J;
  PetscMPIInt rank, size;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  n = 2 * size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatDuplicate(C, MAT_DO_NOT_COPY_VALUES, &B)); /* test that SeqAIJ non-preallocated matrices can be duplicated */
  PetscCall(MatDestroy(&C));
  C = B;
  for (i = 0; i < m; i++) {
    for (j = 2 * rank; j < 2 * rank + 2; j++) {
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
  for (i = 0; i < m; i++) {
    for (j = 2 * rank; j < 2 * rank + 2; j++) {
      v  = 1.0;
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
      v = -4.0;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &Ii, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
