
static char help[] = "Tests automatic allocation of matrix storage space.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         C;
  PetscInt    i, j, m = 3, n = 3, Ii, J;
  PetscScalar v;
  MatInfo     info;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* create the matrix for the five point stencil, YET AGAIN */
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
  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatGetInfo(C, MAT_LOCAL, &info));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "matrix nonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n", (PetscInt)info.nz_used, (PetscInt)info.nz_allocated));

  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
