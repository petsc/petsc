
static char help[] = "Tests sequential and parallel MatGetRow() and MatRestoreRow().\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat                C;
  PetscInt           i, j, m = 5, n = 5, Ii, J, nz, rstart, rend;
  PetscMPIInt        rank;
  const PetscInt    *idx;
  PetscScalar        v;
  const PetscScalar *values;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  /* Create the matrix for the five point stencil, YET AGAIN */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
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
  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MatGetOwnershipRange(C, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatGetRow(C, i, &nz, &idx, &values));
    if (rank == 0) {
      for (j = 0; j < nz; j++) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT " %g ", idx[j], (double)PetscRealPart(values[j])));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
    }
    PetscCall(MatRestoreRow(C, i, &nz, &idx, &values));
  }

  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
