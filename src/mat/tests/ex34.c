static char help[] = "Test MatMatMult() and MatTransposeMatMult() for MPIAIJ and MPIDENSE matrices. \n\
                      Sequential part of mpidense matrix allows changes made by MatDenseSetLDA(). \n\n";

#include <petsc.h>

int main(int argc, char ** argv)
{
  Mat            A, B, C, C1;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscInt       i,ia[2] = { 0, 2 }, ja[2] = { 0, 1 }, lda = 4;
  PetscScalar    a[2] = { 1.0, 1.0 }, *data;
  PetscBool      flg;

  ierr = PetscInitialize(&argc, &argv, (char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must use 2 processors");

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetType(A, MATMPIAIJ));
  CHKERRQ(MatSetSizes(A, 1, 1, 2, 2));
  CHKERRQ(MatMPIAIJSetPreallocationCSR(A, ia, ja, a));

  CHKERRQ(PetscCalloc1(4 * lda,&data));
  for (i = 0; i < 4; ++i) data[lda * i] = i * 1.0;

  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, 1, PETSC_DECIDE, 2, 4, data, &B));
  CHKERRQ(MatSetOptionsPrefix(B,"b_"));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatDenseSetLDA(B, lda));

  /* Test MatMatMult() */
  CHKERRQ(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
  CHKERRQ(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));

  CHKERRQ(MatMatMultEqual(A,B,C,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatMatMult() for C");

  /* Test user-provided mpidense matrix product */
  CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&C1));
  CHKERRQ(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C1));
  CHKERRQ(MatMatMultEqual(A,B,C1,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatMatMult() for C1");

  CHKERRQ(MatDestroy(&C1));
  CHKERRQ(MatDestroy(&C));

  /* Test MatTransposeMatMult() */
  CHKERRQ(MatTransposeMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
  CHKERRQ(MatTransposeMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));

  CHKERRQ(MatTransposeMatMultEqual(A,B,C,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatTransposeMatMult()");
  CHKERRQ(MatDestroy(&C));

  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFree(data));

  ierr = PetscFinalize();
  return ierr;
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
