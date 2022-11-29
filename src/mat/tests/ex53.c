
static char help[] = "Tests various routines in MatMPIBAIJ format.\n";

#include <petscmat.h>
#define IMAX 15
int main(int argc, char **args)
{
  Mat                A, B, C, At, Bt;
  PetscViewer        fd;
  char               file[PETSC_MAX_PATH_LEN];
  PetscRandom        rand;
  Vec                xx, yy, s1, s2;
  PetscReal          s1norm, s2norm, rnorm, tol = 1.e-10;
  PetscInt           rstart, rend, rows[2], cols[2], m, n, i, j, M, N, ct, row, ncols1, ncols2, bs;
  PetscMPIInt        rank, size;
  const PetscInt    *cols1, *cols2;
  PetscScalar        vals1[4], vals2[4], v;
  const PetscScalar *v1, *v2;
  PetscBool          flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Check out if MatLoad() works */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Input file not specified");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATBAIJ));
  PetscCall(MatLoad(A, fd));

  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &xx));
  PetscCall(VecSetSizes(xx, m, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(xx));
  PetscCall(VecDuplicate(xx, &s1));
  PetscCall(VecDuplicate(xx, &s2));
  PetscCall(VecDuplicate(xx, &yy));
  PetscCall(MatGetBlockSize(A, &bs));

  /* Test MatNorm() */
  PetscCall(MatNorm(A, NORM_FROBENIUS, &s1norm));
  PetscCall(MatNorm(B, NORM_FROBENIUS, &s2norm));
  rnorm = PetscAbsScalar(s2norm - s1norm) / s2norm;
  if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error: MatNorm_FROBENIUS()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)s1norm, (double)s2norm, bs));
  PetscCall(MatNorm(A, NORM_INFINITY, &s1norm));
  PetscCall(MatNorm(B, NORM_INFINITY, &s2norm));
  rnorm = PetscAbsScalar(s2norm - s1norm) / s2norm;
  if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error: MatNorm_INFINITY()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)s1norm, (double)s2norm, bs));
  PetscCall(MatNorm(A, NORM_1, &s1norm));
  PetscCall(MatNorm(B, NORM_1, &s2norm));
  rnorm = PetscAbsScalar(s2norm - s1norm) / s2norm;
  if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error: MatNorm_NORM_1()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)s1norm, (double)s2norm, bs));

  /* Test MatMult() */
  for (i = 0; i < IMAX; i++) {
    PetscCall(VecSetRandom(xx, rand));
    PetscCall(MatMult(A, xx, s1));
    PetscCall(MatMult(B, xx, s2));
    PetscCall(VecAXPY(s2, -1.0, s1));
    PetscCall(VecNorm(s2, NORM_2, &rnorm));
    if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error: MatMult - Norm2=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)rnorm, bs));
  }

  /* test MatMultAdd() */
  for (i = 0; i < IMAX; i++) {
    PetscCall(VecSetRandom(xx, rand));
    PetscCall(VecSetRandom(yy, rand));
    PetscCall(MatMultAdd(A, xx, yy, s1));
    PetscCall(MatMultAdd(B, xx, yy, s2));
    PetscCall(VecAXPY(s2, -1.0, s1));
    PetscCall(VecNorm(s2, NORM_2, &rnorm));
    if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error: MatMultAdd - Norm2=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)rnorm, bs));
  }

  /* Test MatMultTranspose() */
  for (i = 0; i < IMAX; i++) {
    PetscCall(VecSetRandom(xx, rand));
    PetscCall(MatMultTranspose(A, xx, s1));
    PetscCall(MatMultTranspose(B, xx, s2));
    PetscCall(VecNorm(s1, NORM_2, &s1norm));
    PetscCall(VecNorm(s2, NORM_2, &s2norm));
    rnorm = s2norm - s1norm;
    if (rnorm < -tol || rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error: MatMultTranspose - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)s1norm, (double)s2norm, bs));
  }
  /* Test MatMultTransposeAdd() */
  for (i = 0; i < IMAX; i++) {
    PetscCall(VecSetRandom(xx, rand));
    PetscCall(VecSetRandom(yy, rand));
    PetscCall(MatMultTransposeAdd(A, xx, yy, s1));
    PetscCall(MatMultTransposeAdd(B, xx, yy, s2));
    PetscCall(VecNorm(s1, NORM_2, &s1norm));
    PetscCall(VecNorm(s2, NORM_2, &s2norm));
    rnorm = s2norm - s1norm;
    if (rnorm < -tol || rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error: MatMultTransposeAdd - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)s1norm, (double)s2norm, bs));
  }

  /* Check MatGetValues() */
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatGetSize(A, &M, &N));

  for (i = 0; i < IMAX; i++) {
    /* Create random row numbers ad col numbers */
    PetscCall(PetscRandomGetValue(rand, &v));
    cols[0] = (int)(PetscRealPart(v) * N);
    PetscCall(PetscRandomGetValue(rand, &v));
    cols[1] = (int)(PetscRealPart(v) * N);
    PetscCall(PetscRandomGetValue(rand, &v));
    rows[0] = rstart + (int)(PetscRealPart(v) * m);
    PetscCall(PetscRandomGetValue(rand, &v));
    rows[1] = rstart + (int)(PetscRealPart(v) * m);

    PetscCall(MatGetValues(A, 2, rows, 2, cols, vals1));
    PetscCall(MatGetValues(B, 2, rows, 2, cols, vals2));

    for (j = 0; j < 4; j++) {
      if (vals1[j] != vals2[j]) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d]: Error: MatGetValues rstart = %2" PetscInt_FMT "  row = %2" PetscInt_FMT " col = %2" PetscInt_FMT " val1 = %e val2 = %e bs = %" PetscInt_FMT "\n", rank, rstart, rows[j / 2], cols[j % 2], (double)PetscRealPart(vals1[j]), (double)PetscRealPart(vals2[j]), bs));
      }
    }
  }

  /* Test MatGetRow()/ MatRestoreRow() */
  for (ct = 0; ct < 100; ct++) {
    PetscCall(PetscRandomGetValue(rand, &v));
    row = rstart + (PetscInt)(PetscRealPart(v) * m);
    PetscCall(MatGetRow(A, row, &ncols1, &cols1, &v1));
    PetscCall(MatGetRow(B, row, &ncols2, &cols2, &v2));

    for (i = 0, j = 0; i < ncols1 && j < ncols2; j++) {
      while (cols2[j] != cols1[i]) i++;
      PetscCheck(v1[i] == v2[j], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatGetRow() failed - vals incorrect.");
    }
    PetscCheck(j >= ncols2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatGetRow() failed - cols incorrect");

    PetscCall(MatRestoreRow(A, row, &ncols1, &cols1, &v1));
    PetscCall(MatRestoreRow(B, row, &ncols2, &cols2, &v2));
  }

  /* Test MatConvert() */
  PetscCall(MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &C));

  /* See if MatMult Says both are same */
  for (i = 0; i < IMAX; i++) {
    PetscCall(VecSetRandom(xx, rand));
    PetscCall(MatMult(A, xx, s1));
    PetscCall(MatMult(C, xx, s2));
    PetscCall(VecNorm(s1, NORM_2, &s1norm));
    PetscCall(VecNorm(s2, NORM_2, &s2norm));
    rnorm = s2norm - s1norm;
    if (rnorm < -tol || rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error in MatConvert: MatMult - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)s1norm, (double)s2norm, bs));
  }
  PetscCall(MatDestroy(&C));

  /* Test MatTranspose() */
  PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &At));
  PetscCall(MatTranspose(B, MAT_INITIAL_MATRIX, &Bt));
  for (i = 0; i < IMAX; i++) {
    PetscCall(VecSetRandom(xx, rand));
    PetscCall(MatMult(At, xx, s1));
    PetscCall(MatMult(Bt, xx, s2));
    PetscCall(VecNorm(s1, NORM_2, &s1norm));
    PetscCall(VecNorm(s2, NORM_2, &s2norm));
    rnorm = s2norm - s1norm;
    if (rnorm < -tol || rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Error in MatConvert:MatMult - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n", rank, (double)s1norm, (double)s2norm, bs));
  }
  PetscCall(MatDestroy(&At));
  PetscCall(MatDestroy(&Bt));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&xx));
  PetscCall(VecDestroy(&yy));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 1 -f ${DATAFILESPATH}/matrices/small

   test:
      suffix: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 2 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      suffix: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 4 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 4
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 5 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      suffix: 5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 6 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 6
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 7 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 7
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 8 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      suffix: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 4
      args: -matload_block_size 3 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

TEST*/
