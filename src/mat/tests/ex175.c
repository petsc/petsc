static char help[] = "Tests MatCreateHermitianTranspose().\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         C, C_htransposed, Cht, C_empty, Cht_empty;
  PetscInt    i, j, m = 10, n = 10;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  /* Create a complex non-hermitian matrix */
  PetscCall(MatCreate(PETSC_COMM_SELF, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      v = 0.0 - 1.0 * PETSC_i;
      if (i > j && i - j < 2) PetscCall(MatSetValues(C, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateHermitianTranspose(C, &C_htransposed));

  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatDuplicate(C_htransposed, MAT_COPY_VALUES, &Cht));
  PetscCall(MatConvert(Cht, MATAIJ, MAT_INPLACE_MATRIX, &Cht));
  PetscCall(MatView(Cht, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatDuplicate(C_htransposed, MAT_DO_NOT_COPY_VALUES, &C_empty));
  PetscCall(MatHermitianTransposeGetMat(C_empty, &Cht_empty));
  PetscCall(MatHermitianTranspose(Cht_empty, MAT_INPLACE_MATRIX, &Cht_empty));
  PetscCall(MatView(Cht_empty, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&C_htransposed));
  PetscCall(MatDestroy(&Cht));
  PetscCall(MatDestroy(&C_empty));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: complex

   test:
     output_file: output/ex175.out

TEST*/
