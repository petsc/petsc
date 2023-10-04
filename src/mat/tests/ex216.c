#include <petsc.h>

static char help[] = "Tests for MatEliminateZeros().\n\n";

int main(int argc, char **args)
{
  Mat       A, B, C, D, E;
  PetscInt  M = 40, bs = 2;
  PetscReal threshold = 1.2;
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-threshold", &threshold, NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, M, NULL, &D));
  PetscCall(MatSetRandom(D, NULL));
  PetscCall(MatTranspose(D, MAT_INITIAL_MATRIX, &A));
  PetscCall(MatAXPY(D, 1.0, A, SAME_NONZERO_PATTERN));
  PetscCall(MatDestroy(&A));
  PetscCall(MatSetBlockSize(D, bs));
  PetscCall(MatDuplicate(D, MAT_COPY_VALUES, &E));
  PetscCall(MatViewFromOptions(D, NULL, "-input_dense"));
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(MatConvert(D, MATAIJ, MAT_INITIAL_MATRIX, &A));
    PetscCall(MatConvert(D, MATBAIJ, MAT_INITIAL_MATRIX, &B));
    PetscCall(MatConvert(D, MATSBAIJ, MAT_INITIAL_MATRIX, &C));
    if (i == 0) { // filtering, elimination, but no compression
      PetscCall(MatViewFromOptions(A, NULL, "-input_aij"));
      PetscCall(MatViewFromOptions(B, NULL, "-input_baij"));
      PetscCall(MatViewFromOptions(C, NULL, "-input_sbaij"));
      PetscCall(MatFilter(D, threshold, PETSC_FALSE, PETSC_FALSE));
      PetscCall(MatFilter(A, threshold, PETSC_FALSE, PETSC_FALSE));
      PetscCall(MatEliminateZeros(A, PETSC_TRUE));
      PetscCall(MatFilter(B, threshold, PETSC_FALSE, PETSC_FALSE));
      PetscCall(MatEliminateZeros(B, PETSC_TRUE));
      PetscCall(MatFilter(C, threshold, PETSC_FALSE, PETSC_FALSE));
      PetscCall(MatEliminateZeros(C, PETSC_TRUE));
    } else { // filtering, elimination, and compression
      PetscCall(MatFilter(D, threshold, PETSC_TRUE, PETSC_FALSE));
      PetscCall(MatFilter(A, threshold, PETSC_TRUE, PETSC_FALSE));
      PetscCall(MatFilter(B, threshold, PETSC_TRUE, PETSC_FALSE));
      PetscCall(MatFilter(C, threshold, PETSC_TRUE, PETSC_FALSE));
    }
    PetscCall(MatViewFromOptions(D, NULL, "-output_dense"));
    PetscCall(MatViewFromOptions(A, NULL, "-output_aij"));
    PetscCall(MatViewFromOptions(B, NULL, "-output_baij"));
    PetscCall(MatViewFromOptions(C, NULL, "-output_sbaij"));
    PetscCall(MatMultEqual(D, A, 10, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "A != D");
    PetscCall(MatMultEqual(D, B, 10, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "B != D");
    PetscCall(MatMultEqual(D, C, 10, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "C != D");
    PetscCall(MatDestroy(&C));
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&D));
    D = E;
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1 2}}
      output_file: output/empty.out

TEST*/
