static char help[] = "Tests assembly of a matrix from another matrix's hash table.\n\n";

#include <petscmat.h>

PetscErrorCode SetValues(Mat A)
{
  PetscInt    m, n, i, j;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(A, &m, &n));
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      v = 10.0 * i + j + 1;
      PetscCall(MatSetValues(A, 1, &i, 1, &j, &v, ADD_VALUES));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateAndViewB(Mat A)
{
  Mat B;

  PetscFunctionBeginUser;
  /* Create B */
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
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* ------- Set values in A --------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 1, 1, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(SetValues(A));
  PetscCall(CreateAndViewB(A));
  PetscCall(AssembleAndViewA(A));

  PetscCall(MatResetHash(A));

  PetscCall(SetValues(A));
  PetscCall(CreateAndViewB(A));
  PetscCall(AssembleAndViewA(A));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: seq
      args: -mat_type seqaij
      filter: grep -v "Mat Object"

   test:
      suffix: mpi
      args: -mat_type mpiaij
      nsize: 4
      filter: grep -v "Mat Object"

TEST*/
