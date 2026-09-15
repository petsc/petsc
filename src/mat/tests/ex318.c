static const char help[] = "Tests MatDenseGetColumnVec() and friends on dense matrices created from a VecType, and on the Mats a MatProduct derives from them\n\n";

#include <petscmat.h>

/* The Mat a product creates only keeps the VecType when it is the same kind of Mat as the one it is built from */
static PetscErrorCode CheckProductVecType(Mat C, Mat A, const char *what)
{
  MatType   amtype, cmtype;
  VecType   avtype, cvtype;
  PetscBool same;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(A, &amtype));
  PetscCall(MatGetType(C, &cmtype));
  PetscCall(PetscStrcmp(amtype, cmtype, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatGetVecType(A, &avtype));
  PetscCall(MatGetVecType(C, &cvtype));
  PetscCall(PetscStrcmp(avtype, cvtype, &same));
  PetscCheck(same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "%s has VecType %s, expected %s", what, cvtype, avtype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A dense Mat of the MatType of A, which has the default VecType of that MatType */
static PetscErrorCode CreateDenseDefaultVecType(Mat A, PetscInt M, PetscInt N, Mat *B)
{
  MatType mtype;

  PetscFunctionBeginUser;
  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(*B, mtype));
  PetscCall(MatSetUp(*B));
  PetscCall(MatZeroEntries(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat                A, C, S, S2, C2, C3, C4, C5, C6, D, E, F;
  Vec                v, w;
  char               vtype[64] = VECSTANDARD;
  VecType            avtype;
  PetscBool          same;
  PetscInt           M = 9, N = 3, lda, rstart, rend, i, j;
  PetscReal          norm;
  PetscScalar        sum;
  const PetscScalar *array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-vec_type", vtype, sizeof(vtype), NULL));
  /* A VECKOKKOS type gives a MATDENSECUDA/MATDENSEHIP with VECKOKKOS vectors when the Kokkos backend runs on that device */
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, PETSC_DECIDE, PETSC_DECIDE, M, N, PETSC_DECIDE, NULL, &A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

  /* Write column j through its column vector, then scale it through a read/write column vector: A(:, j) = 2 (j + 1) */
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVecWrite(A, j, &v));
    PetscCall(VecSet(v, (PetscScalar)(j + 1)));
    PetscCall(MatDenseRestoreColumnVecWrite(A, j, &v));
  }
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVec(A, j, &v));
    PetscCall(VecScale(v, 2.0));
    PetscCall(MatDenseRestoreColumnVec(A, j, &v));
  }

  /* Check through read-only column vectors, and independently through the matrix itself */
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVecRead(A, j, &v));
    PetscCall(VecSum(v, &sum));
    PetscCheck(PetscAbsScalar(sum - 2.0 * (j + 1) * M) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " read back through MatDenseGetColumnVecRead() sums to %g, expected %g", j, (double)PetscRealPart(sum), 2.0 * (j + 1) * M);
    PetscCall(MatDenseRestoreColumnVecRead(A, j, &v));
  }
  PetscCall(MatNorm(A, NORM_INFINITY, &norm));
  PetscCheck(PetscAbsReal(norm - N * (N + 1)) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatNorm() gives %g, expected %g", (double)norm, (double)(N * (N + 1)));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(MatDenseGetArrayRead(A, &array));
  for (j = 0; j < N; j++) {
    for (i = 0; i < rend - rstart; i++)
      PetscCheck(array[i + j * lda] == 2.0 * (j + 1), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Entry (%" PetscInt_FMT ", %" PetscInt_FMT ") is %g, expected %g", rstart + i, j, (double)PetscRealPart(array[i + j * lda]), 2.0 * (j + 1));
  }
  PetscCall(MatDenseRestoreArrayRead(A, &array));

  /* The Mat a MatProduct creates must have the VecType of the Mat it is built from */
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, rend - rstart, rend - rstart, M, M, 3.0, &S));
  PetscCall(MatMatMult(S, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatGetVecType(A, &avtype));
  PetscCall(CheckProductVecType(C, A, "The Mat the product created"));

  /* C is 3 A, so the columns cancel, which needs both column Vecs to be of the same type */
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVec(C, j, &v));
    PetscCall(MatDenseGetColumnVecRead(A, j, &w));
    PetscCall(VecAXPY(v, -3.0, w));
    PetscCall(VecNorm(v, NORM_INFINITY, &norm));
    PetscCall(MatDenseRestoreColumnVecRead(A, j, &w));
    PetscCall(MatDenseRestoreColumnVec(C, j, &v));
    PetscCheck(norm < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " of the product differs from three times the column it was built from by %g", j, (double)norm);
  }

  /* An AIJ times dense product takes a different symbolic route; it only keeps the VecType when the Mat it
     creates is the same kind of Mat as the block, which is not the case for a host AIJ and a device block */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &S2));
  PetscCall(MatSetSizes(S2, rend - rstart, rend - rstart, M, M));
  PetscCall(MatSetType(S2, MATAIJ));
  PetscCall(MatSetUp(S2));
  for (i = rstart; i < rend; i++) PetscCall(MatSetValue(S2, i, i, 3.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(S2, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S2, MAT_FINAL_ASSEMBLY));
  PetscCall(MatMatMult(S2, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C2));
  PetscCall(CheckProductVecType(C2, A, "The Mat the AIJ product created"));

  /* and the two dense times dense products, which take a further symbolic route */
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, PETSC_DECIDE, PETSC_DECIDE, N, N, PETSC_DECIDE, NULL, &D));
  PetscCall(MatZeroEntries(D));
  PetscCall(MatMatMult(A, D, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C3));
  PetscCall(CheckProductVecType(C3, A, "The Mat the dense product created"));
  PetscCall(MatMatTransposeMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C4));
  PetscCall(CheckProductVecType(C4, A, "The Mat the dense transpose product created"));

  /* and a dense times AIJ product, which is not available for a sequential MATDENSECUDA or MATDENSEHIP */
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &same, MATSEQDENSE, MATMPIDENSE, ""));
  if (same) {
    PetscCall(MatMatMult(C4, S2, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C5));
    PetscCall(CheckProductVecType(C5, A, "The Mat the dense times AIJ product created"));
    PetscCall(MatDestroy(&C5));
  }

  /* with dense operands of different VecType the Mat a product creates takes that of A, whatever the number of ranks */
  PetscCall(CreateDenseDefaultVecType(A, N, N, &E));
  PetscCall(CreateDenseDefaultVecType(A, M, N, &F));
  PetscCall(MatMatMult(A, E, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C6));
  PetscCall(CheckProductVecType(C6, A, "The Mat the dense product with operands of different VecType created"));
  PetscCall(MatDestroy(&C6));
  PetscCall(MatMatTransposeMult(A, F, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C6));
  PetscCall(CheckProductVecType(C6, A, "The Mat the dense transpose product with operands of different VecType created"));
  PetscCall(MatDestroy(&C6));
  PetscCall(MatTransposeMatMult(A, F, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C6));
  PetscCall(CheckProductVecType(C6, A, "The Mat the transpose dense product with operands of different VecType created"));
  PetscCall(MatDestroy(&C6));

  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&C4));
  PetscCall(MatDestroy(&C3));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C2));
  PetscCall(MatDestroy(&S2));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}
    test:
      suffix: standard
      args: -vec_type standard
    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda
    test:
      suffix: hip
      requires: hip
      args: -vec_type hip
    test:
      suffix: kokkos
      requires: kokkos_kernels !sycl
      args: -vec_type kokkos

TEST*/
