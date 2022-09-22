static char help[] = "Test MatMatMult() and MatTransposeMatMult() for MPIAIJ and MPIDENSE matrices. \n\
                      Sequential part of mpidense matrix allows changes made by MatDenseSetLDA(). \n\n";

#include <petsc.h>

int main(int argc, char **argv)
{
  Mat         A, B, C, C1;
  PetscMPIInt size;
  PetscInt    i, ia[2] = {0, 2}, ja[2] = {0, 1}, lda = 4;
  PetscScalar a[2] = {1.0, 1.0}, *data;
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must use 2 processors");

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetSizes(A, 1, 1, 2, 2));
  PetscCall(MatMPIAIJSetPreallocationCSR(A, ia, ja, a));

  PetscCall(PetscCalloc1(4 * lda, &data));
  for (i = 0; i < 4; ++i) data[lda * i] = i * 1.0;

  PetscCall(MatCreateDense(PETSC_COMM_WORLD, 1, PETSC_DECIDE, 2, 4, data, &B));
  PetscCall(MatSetOptionsPrefix(B, "b_"));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatDenseSetLDA(B, lda));

  /* Test MatMatMult() */
  PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
  PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));

  PetscCall(MatMatMultEqual(A, B, C, 10, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in MatMatMult() for C");

  /* Test user-provided mpidense matrix product */
  PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &C1));
  PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C1));
  PetscCall(MatMatMultEqual(A, B, C1, 10, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in MatMatMult() for C1");

  PetscCall(MatDestroy(&C1));
  PetscCall(MatDestroy(&C));

  /* Test MatTransposeMatMult() */
  PetscCall(MatTransposeMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
  PetscCall(MatTransposeMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));

  PetscCall(MatTransposeMatMultEqual(A, B, C, 10, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in MatTransposeMatMult()");
  PetscCall(MatDestroy(&C));

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(data));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      output_file: output/ex34.out

   test:
      suffix: 1_cuda
      requires: cuda
      nsize: 2
      args: -b_mat_type mpidensecuda
      output_file: output/ex34.out
TEST*/
