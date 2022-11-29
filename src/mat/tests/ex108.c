static char help[] = "Testing MatCreateSeqBAIJWithArrays() and MatCreateSeqSBAIJWithArrays().\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat             A, B, As;
  const PetscInt *ai, *aj;
  PetscInt        i, j, k, nz, n, asi[] = {0, 2, 3, 4, 6, 7};
  PetscInt        asj[] = {0, 4, 1, 2, 3, 4, 4};
  PetscScalar     asa[7], *aa;
  PetscRandom     rctx;
  PetscMPIInt     size;
  PetscBool       flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Create a aij matrix for checking */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 5, 5, 2, NULL, &A));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));

  k = 0;
  for (i = 0; i < 5; i++) {
    nz = asi[i + 1] - asi[i]; /* length of i_th row of A */
    for (j = 0; j < nz; j++) {
      PetscCall(PetscRandomGetValue(rctx, &asa[k]));
      PetscCall(MatSetValues(A, 1, &i, 1, &asj[k], &asa[k], INSERT_VALUES));
      PetscCall(MatSetValues(A, 1, &i, 1, &asj[k], &asa[k], INSERT_VALUES));
      if (i != asj[k]) { /* insert symmetric entry */
        PetscCall(MatSetValues(A, 1, &asj[k], 1, &i, &asa[k], INSERT_VALUES));
      }
      k++;
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Create a baij matrix using MatCreateSeqBAIJWithArrays() */
  PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &n, &ai, &aj, &flg));
  PetscCall(MatSeqAIJGetArray(A, &aa));
  /* WARNING: This sharing is dangerous if either A or B is later assembled */
  PetscCall(MatCreateSeqBAIJWithArrays(PETSC_COMM_SELF, 1, 5, 5, (PetscInt *)ai, (PetscInt *)aj, aa, &B));
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &n, &ai, &aj, &flg));
  PetscCall(MatMultEqual(A, B, 10, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "MatMult(A,B) are NOT equal");

  /* Create a sbaij matrix using MatCreateSeqSBAIJWithArrays() */
  PetscCall(MatCreateSeqSBAIJWithArrays(PETSC_COMM_SELF, 1, 5, 5, asi, asj, asa, &As));
  PetscCall(MatMultEqual(A, As, 10, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "MatMult(A,As) are NOT equal");

  /* Free spaces */
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&As));
  PetscCall(PetscFinalize());
  return 0;
}
