static const char help[] = "Tests MatDenseGetColumnVec() and friends on dense matrices created from a VecType, including device dense matrices whose vectors are VECKOKKOS\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat                A;
  Vec                v;
  char               vtype[64] = VECSTANDARD;
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
    # On a CUDA or HIP build this is a MATDENSECUDA or MATDENSEHIP with VECKOKKOS column vectors
    test:
      suffix: kokkos
      requires: kokkos_kernels !sycl
      args: -vec_type kokkos

TEST*/
