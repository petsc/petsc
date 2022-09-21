static char help[] = "Testing MatCreateMPIMatConcatenateSeqMat().\n\n";

#include <petscmat.h>
int main(int argc, char **argv)
{
  Mat         seqmat, mpimat;
  PetscMPIInt rank;
  PetscScalar value[3], *vals;
  PetscInt    i, col[3], n = 5, bs = 1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));

  /* Create seqaij matrices of size (n+rank) by n */
  PetscCall(MatCreate(PETSC_COMM_SELF, &seqmat));
  PetscCall(MatSetSizes(seqmat, (n + rank) * bs, PETSC_DECIDE, PETSC_DECIDE, n * bs));
  PetscCall(MatSetFromOptions(seqmat));
  PetscCall(MatSeqAIJSetPreallocation(seqmat, 3 * bs, NULL));
  PetscCall(MatSeqBAIJSetPreallocation(seqmat, bs, 3, NULL));

  if (bs == 1) {
    value[0] = -1.0;
    value[1] = 2.0;
    value[2] = -1.0;
    for (i = 1; i < n - 1; i++) {
      col[0] = i - 1;
      col[1] = i;
      col[2] = i + 1;
      PetscCall(MatSetValues(seqmat, 1, &i, 3, col, value, INSERT_VALUES));
    }
    i      = n - 1;
    col[0] = n - 2;
    col[1] = n - 1;
    PetscCall(MatSetValues(seqmat, 1, &i, 2, col, value, INSERT_VALUES));

    i        = 0;
    col[0]   = 0;
    col[1]   = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    PetscCall(MatSetValues(seqmat, 1, &i, 2, col, value, INSERT_VALUES));
  } else {
    PetscInt *rows, *cols, j;
    PetscCall(PetscMalloc3(bs * bs, &vals, bs, &rows, bs, &cols));
    /* diagonal blocks */
    for (i = 0; i < bs * bs; i++) vals[i] = 2.0;
    for (i = 0; i < n * bs; i += bs) {
      for (j = 0; j < bs; j++) {
        rows[j] = i + j;
        cols[j] = i + j;
      }
      PetscCall(MatSetValues(seqmat, bs, rows, bs, cols, vals, INSERT_VALUES));
    }
    /* off-diagonal blocks */
    for (i = 0; i < bs * bs; i++) vals[i] = -1.0;
    for (i = 0; i < (n - 1) * bs; i += bs) {
      for (j = 0; j < bs; j++) {
        rows[j] = i + j;
        cols[j] = i + bs + j;
      }
      PetscCall(MatSetValues(seqmat, bs, rows, bs, cols, vals, INSERT_VALUES));
    }

    PetscCall(PetscFree3(vals, rows, cols));
  }
  PetscCall(MatAssemblyBegin(seqmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(seqmat, MAT_FINAL_ASSEMBLY));
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] seqmat:\n", rank));
    PetscCall(MatView(seqmat, PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD, seqmat, PETSC_DECIDE, MAT_INITIAL_MATRIX, &mpimat));
  PetscCall(MatCreateMPIMatConcatenateSeqMat(PETSC_COMM_WORLD, seqmat, PETSC_DECIDE, MAT_REUSE_MATRIX, &mpimat));
  PetscCall(MatView(mpimat, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&seqmat));
  PetscCall(MatDestroy(&mpimat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

   test:
      suffix: 2
      nsize: 3
      args: -mat_type baij

   test:
      suffix: 3
      nsize: 3
      args: -mat_type baij -bs 2

TEST*/
