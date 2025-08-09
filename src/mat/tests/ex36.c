static char help[] = "Tests assembly of a matrix from another matrix's hash table.\n\n";

#include <petscmat.h>

PetscErrorCode SetValues(Mat A, PetscBool zero, PetscBool insertvals)
{
  PetscInt    m, n, i, j;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(A, &m, &n));
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      v = zero ? 0.0 : 10.0 * i + j + 1;
      PetscCall(MatSetValues(A, 1, &i, 1, &j, &v, insertvals ? INSERT_VALUES : ADD_VALUES));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateAndViewB(Mat A)
{
  Mat B;

  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatCopyHashToXAIJ(A, B));
  PetscCall(MatView(B, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssembleAndViewA(Mat A)
{
  PetscFunctionBeginUser;
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat       A, T;
  PetscInt  N, n, m;
  PetscBool zero = PETSC_FALSE, ignorezero = PETSC_FALSE, insertvals = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero", &zero, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ignorezero", &ignorezero, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-insertvals", &insertvals, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &T));
  PetscCall(MatSetSizes(T, 1, 1, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(T));
  PetscCall(MatGetSize(T, NULL, &N));
  PetscCall(MatGetLocalSize(T, &m, &n));
  PetscCall(MatSeqAIJSetPreallocation(T, N, NULL));
  PetscCall(MatMPIAIJSetPreallocation(T, n, NULL, N - n, NULL));
  PetscCall(MatSetOption(T, MAT_IGNORE_ZERO_ENTRIES, ignorezero));
  PetscCall(MatSetUp(T));
  PetscCall(SetValues(T, zero, insertvals));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DEBUG T\n"));
  PetscCall(AssembleAndViewA(T));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 1, 1, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, ignorezero));
  PetscCall(MatSetUp(A));

  PetscCall(SetValues(A, zero, insertvals));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DEBUG B\n"));
  PetscCall(CreateAndViewB(A));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DEBUG A\n"));
  PetscCall(AssembleAndViewA(A));

  PetscCall(MatResetHash(A));
  /* need to reset the option for MPIAIJ */
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, ignorezero));

  PetscCall(SetValues(A, zero, insertvals));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DEBUG B\n"));
  PetscCall(CreateAndViewB(A));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DEBUG A\n"));
  PetscCall(AssembleAndViewA(A));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&T));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: seq
      diff_args: -j
      args: -mat_type seqaij
      filter: grep -v "Mat Object"

   test:
      suffix: mpi
      diff_args: -j
      args: -mat_type mpiaij
      nsize: 4
      filter: grep -v "Mat Object"

   test:
      diff_args: -j
      suffix: seq_ignore
      args: -mat_type seqaij -zero {{0 1}separate output} -ignorezero {{0 1}separate output} -insertvals {{0 1}separate output}
      filter: grep -v "Mat Object"

   test:
      diff_args: -j
      suffix: mpi
      args: -mat_type mpiaij -zero {{0 1}separate output} -ignorezero {{0 1}separate output} -insertvals {{0 1}separate output}
      nsize: 4
      filter: grep -v "Mat Object"

TEST*/
