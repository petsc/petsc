static char help[] = "Test MatDenseGetSubMatrix() on a CUDA matrix and MatSetInf() on dense matrices.\n";

#include <petscmat.h>

/*
   Every entry of the local block must be positive infinity, while the padding between two columns, which is only present when the leading dimension is larger than the number of local rows, must be left alone
*/
static PetscErrorCode CheckInf(Mat A)
{
  const PetscScalar *a;
  PetscInt           i, j, m, N, lda;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatGetSize(A, NULL, &N));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(MatDenseGetArrayRead(A, &a));
  for (j = 0; j < N; j++) {
    for (i = 0; i < m; i++) PetscCheck(PetscIsInfReal(PetscRealPart(a[i + j * lda])) && PetscRealPart(a[i + j * lda]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Entry (%" PetscInt_FMT ",%" PetscInt_FMT ") is not positive infinity", i, j);
    for (i = m; i < lda; i++) PetscCheck(a[i + j * lda] == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Padding entry (%" PetscInt_FMT ",%" PetscInt_FMT ") was overwritten", i, j);
  }
  PetscCall(MatDenseRestoreArrayRead(A, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Device dense Mat with n local rows, n global columns and the given leading dimension. MatCreateDenseCUDA() preallocates the storage before MatDenseSetLDA() can take effect, hence the MatSetType() path below
*/
static PetscErrorCode CreateDenseCUDA(PetscInt n, PetscInt lda, Mat *A)
{
  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, n, PETSC_DECIDE, PETSC_DETERMINE, n));
  PetscCall(MatSetType(*A, MATDENSECUDA));
  PetscCall(MatDenseSetLDA(*A, lda));
  PetscCall(MatSetUp(*A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   MatSetInf() on a dense Mat whose leading dimension matches the number of local rows and on one whose leading dimension is larger, on the host or on the device
*/
static PetscErrorCode TestSetInf(PetscInt n, PetscInt lda, PetscBool cuda)
{
  Mat          A;
  PetscScalar *data = NULL;

  PetscFunctionBeginUser;
  if (cuda) PetscCall(CreateDenseCUDA(n, n, &A));
  else PetscCall(MatCreateDense(PETSC_COMM_WORLD, n, PETSC_DECIDE, PETSC_DETERMINE, n, NULL, &A));
  PetscCall(MatZeroEntries(A));
  PetscCall(MatSetInf(A));
  PetscCall(CheckInf(A));
  PetscCall(MatDestroy(&A));

  if (cuda) {
    PetscCall(CreateDenseCUDA(n, lda, &A));
    /* the values are put on the device so that MatSetInf() has to invalidate them, as when it is called on the block of solutions of a KSPMatSolve() which has not converged */
    PetscCall(MatZeroEntries(A));
  } else {
    PetscCall(PetscCalloc1(lda * n, &data));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, n, PETSC_DECIDE, PETSC_DETERMINE, n, data, &A));
    PetscCall(MatDenseSetLDA(A, lda));
  }
  PetscCall(MatSetInf(A));
  PetscCall(CheckInf(A));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat          A, B;
  PetscScalar *b;
  PetscInt     n = 4, lda = 5, i, k;
  PetscBool    cuda = PETSC_FALSE, set_inf = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-lda", &lda, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cuda", &cuda, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-set_inf", &set_inf, NULL));
  PetscCheck(lda >= n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "lda %" PetscInt_FMT " < n %" PetscInt_FMT, lda, n);

  if (set_inf) PetscCall(TestSetInf(n, lda, cuda));
  else {
#if PetscDefined(HAVE_CUDA)
    if (cuda) PetscCall(MatCreateSeqDenseCUDA(PETSC_COMM_SELF, lda, n, NULL, &A));
    else
#endif
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, lda, n, NULL, &A));

    for (k = 0; k < 3; k++) {
      PetscCall(MatDenseGetSubMatrix(A, 0, n, 0, n, &B));
      PetscCall(MatDenseGetArray(B, &b));
      for (i = 0; i < n; i++) {
        b[i + i * lda] = 2.0 * (i + 1);
        if (i > 0) b[i + (i - 1) * lda] = (PetscReal)(k + 1);
      }
      PetscCall(MatDenseRestoreArray(B, &b));
      PetscCall(MatDenseRestoreSubMatrix(A, &B));
      PetscCall(MatView(A, NULL));
    }

    PetscCall(MatDestroy(&A));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/ex257_1.out
      diff_args: -j
      test:
         suffix: 1
      test:
         suffix: 1_cuda
         args: -cuda
         requires: cuda
         filter: sed -e "s/seqdensecuda/seqdense/"

   test:
      suffix: set_inf
      nsize: {{1 2}}
      args: -set_inf
      output_file: output/empty.out

   test:
      suffix: set_inf_cuda
      nsize: {{1 2}}
      args: -set_inf -cuda
      requires: cuda
      output_file: output/empty.out

TEST*/
