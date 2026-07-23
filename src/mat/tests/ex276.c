static char help[] = "Tests MatGetValues() with and without MAT_ROW_ORIENTED\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A, B, C, D, E;
  PetscViewer  viewer;
  PetscInt     rstart, rend, *rows, *cols;
  PetscScalar *valuesA, *valuesB, *valuesC, *valuesD, *valuesE;
  PetscBool    flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "${PETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64", FILE_MODE_READ, &viewer));
  PetscCall(MatLoad(A, viewer));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatConvert(A, MATBAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatConvert(A, MATSBAIJ, MAT_INITIAL_MATRIX, &C));
  PetscCall(MatConvert(A, MATSELL, MAT_INITIAL_MATRIX, &D));
  PetscCall(MatConvert(A, MATDENSE, MAT_INITIAL_MATRIX, &E));

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(PetscMalloc2((rend - rstart - 2), &rows, 6, &cols));
  for (PetscInt i = rstart + 1; i < rend - 1; i++) rows[i - rstart - 1] = i;
  cols[0] = -1;
  cols[1] = 1;
  cols[2] = rstart;
  cols[3] = rstart + 1;
  cols[4] = rstart + 2;
  cols[5] = rstart + 3;

  PetscCall(PetscMalloc5((rend - rstart - 2) * 6, &valuesA, (rend - rstart - 2) * 6, &valuesB, (rend - rstart - 2) * 6, &valuesC, (rend - rstart - 2) * 6, &valuesD, (rend - rstart - 2) * 6, &valuesE));

  PetscCall(PetscArrayzero(valuesA, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(A, rend - rstart - 2, rows, 6, cols, valuesA));
  PetscCall(PetscArrayzero(valuesB, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(B, rend - rstart - 2, rows, 6, cols, valuesB));
  PetscCall(PetscArrayzero(valuesC, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(C, rend - rstart - 2, rows, 6, cols, valuesC));
  PetscCall(PetscArrayzero(valuesD, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(D, rend - rstart - 2, rows, 6, cols, valuesD));
  PetscCall(PetscArrayzero(valuesE, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(E, rend - rstart - 2, rows, 6, cols, valuesE));

  PetscCall(PetscArraycmp(valuesA, valuesB, 6 * (rend - rstart - 2), &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error unexpected values from MatGetValues() valuesB");
  // cannot compare with C since MatGetValues() on MATSBAIJ does not return all the entries
  PetscCall(PetscArraycmp(valuesA, valuesD, 6 * (rend - rstart - 2), &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error unexpected values from MatGetValues() valuesD");
  PetscCall(PetscArraycmp(valuesA, valuesE, 6 * (rend - rstart - 2), &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error unexpected values from MatGetValues() valuesE");

  PetscCall(MatSetOption(A, MAT_ROW_ORIENTED, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_ROW_ORIENTED, PETSC_FALSE));
  PetscCall(MatSetOption(C, MAT_ROW_ORIENTED, PETSC_FALSE));
  PetscCall(MatSetOption(D, MAT_ROW_ORIENTED, PETSC_FALSE));
  PetscCall(MatSetOption(E, MAT_ROW_ORIENTED, PETSC_FALSE));

  PetscCall(PetscArrayzero(valuesA, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(A, rend - rstart - 2, rows, 6, cols, valuesA));
  PetscCall(PetscArrayzero(valuesB, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(B, rend - rstart - 2, rows, 6, cols, valuesB));
  PetscCall(PetscArrayzero(valuesC, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(C, rend - rstart - 2, rows, 6, cols, valuesC));
  PetscCall(PetscArrayzero(valuesD, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(D, rend - rstart - 2, rows, 6, cols, valuesD));
  PetscCall(PetscArrayzero(valuesE, 6 * (rend - rstart - 2)));
  PetscCall(MatGetValues(E, rend - rstart - 2, rows, 6, cols, valuesE));

  PetscCall(PetscArraycmp(valuesA, valuesB, 6 * (rend - rstart - 2), &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error unexpected values from MatGetValues() MAT_ROW_ORIENTED false valuesB");
  // cannot compare with C since MatGetValues() on MATSBAIJ does not return all the entries
  PetscCall(PetscArraycmp(valuesA, valuesD, 6 * (rend - rstart - 2), &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error unexpected values from MatGetValues() MAT_ROW_ORIENTED false valuesD");
  PetscCall(PetscArraycmp(valuesA, valuesE, 6 * (rend - rstart - 2), &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error unexpected values from MatGetValues() MAT_ROW_ORIENTED false valuesE");

  PetscCall(PetscFree2(rows, cols));
  PetscCall(PetscFree5(valuesA, valuesB, valuesC, valuesD, valuesE));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&E));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/empty.out

   test:
      suffix: 2
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      nsize: 2
      output_file: output/empty.out

TEST*/
