static const char help[] = "Tests MatGetDiagonal().\n\n";

#include <petscmat.h>

static PetscErrorCode CheckDiagonal(Mat A, Vec diag, PetscScalar dval)
{
  static PetscBool   first_time = PETSC_TRUE;
  const PetscReal    rtol = 1e-10, atol = PETSC_SMALL;
  PetscInt           rstart, rend, n;
  const PetscScalar *arr;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  // If matrix is AIJ, MatSetRandom() will have randomly choosen the locations of nonzeros,
  // which may not be on the diagonal. So a reallocation is not necessarily a bad thing here.
  if (first_time) PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  for (PetscInt i = rstart; i < rend; ++i) PetscCall(MatSetValue(A, i, i, dval, INSERT_VALUES));
  if (first_time) {
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    first_time = PETSC_FALSE;
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A, NULL, "-mat_view_assembled"));

  PetscCall(VecSetRandom(diag, NULL));
  PetscCall(MatGetDiagonal(A, diag));
  PetscCall(VecViewFromOptions(diag, NULL, "-diag_vec_view"));

  PetscCall(VecGetLocalSize(diag, &n));
  PetscCall(VecGetArrayRead(diag, &arr));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscScalar lhs = arr[i];

    if (!PetscIsCloseAtTolScalar(lhs, dval, rtol, atol)) {
      const PetscReal lhs_r  = PetscRealPart(lhs);
      const PetscReal lhs_i  = PetscImaginaryPart(lhs);
      const PetscReal dval_r = PetscRealPart(dval);
      const PetscReal dval_i = PetscImaginaryPart(dval);

      PetscCheck(PetscIsCloseAtTol(lhs_r, dval_r, rtol, atol), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Real component actual[%" PetscInt_FMT "] %g != expected[%" PetscInt_FMT "] %g", i, (double)lhs_r, i, (double)dval_r);
      PetscCheck(PetscIsCloseAtTol(lhs_i, dval_i, rtol, atol), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Imaginary component actual[%" PetscInt_FMT "] %g != expected[%" PetscInt_FMT "] %g", i, (double)lhs_i, i, (double)dval_i);
    }
  }
  PetscCall(VecRestoreArrayRead(diag, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeMatrix(Mat A)
{
  MatType mtype;
  char   *tmp;

  PetscFunctionBegin;
  PetscCall(MatSetUp(A));
  PetscCall(MatGetType(A, &mtype));
  PetscCall(PetscStrstr(mtype, "aij", &tmp));
  if (tmp) {
    PetscInt  rows, cols, diag_nnz, offdiag_nnz;
    PetscInt *dnnz, *onnz;

    // is some kind of AIJ
    PetscCall(MatGetLocalSize(A, &rows, &cols));
    // at least 3 nonzeros in diagonal block
    diag_nnz = PetscMin(cols, 3);
    // leave at least 3 *zeros* per row
    offdiag_nnz = PetscMax(cols - diag_nnz - 3, 0);
    PetscCall(PetscMalloc2(rows, &dnnz, rows, &onnz));
    for (PetscInt i = 0; i < rows; ++i) {
      dnnz[i] = diag_nnz;
      onnz[i] = offdiag_nnz;
    }
    PetscCall(MatXAIJSetPreallocation(A, PETSC_DECIDE, dnnz, onnz, NULL, NULL));
    PetscCall(PetscFree2(dnnz, onnz));
  }

  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatViewFromOptions(A, NULL, "-mat_view_setup"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat A;
  Vec diag;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 10, 10, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));

  PetscCall(InitializeMatrix(A));

  PetscCall(MatCreateVecs(A, &diag, NULL));

  PetscCall(CheckDiagonal(A, diag, 0.0));
  PetscCall(CheckDiagonal(A, diag, 1.0));
  PetscCall(CheckDiagonal(A, diag, 2.0));

  PetscCall(VecDestroy(&diag));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: ./output/empty.out
    nsize: {{1 2}}
    suffix: dense
    test:
      suffix: standard
      args: -mat_type dense
    test:
      suffix: cuda
      requires: cuda
      args: -mat_type densecuda
    test:
      suffix: hip
      requires: hip
      args: -mat_type densehip

  testset:
    output_file: ./output/empty.out
    nsize: {{1 2}}
    suffix: aij
    test:
      suffix: standard
      args: -mat_type aij
    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse
    test:
      suffix: hip
      requires: hip
      args: -mat_type aijhipsparse
    test:
      suffix: kokkos
      requires: kokkos, kokkos_kernels
      args: -mat_type aijkokkos

TEST*/
