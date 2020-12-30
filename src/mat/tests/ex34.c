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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must use 2 processors");

  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetType(A, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A, 1, 1, 2, 2);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocationCSR(A, ia, ja, a);CHKERRQ(ierr);

  ierr = PetscCalloc1(4 * lda,&data);CHKERRQ(ierr);
  for (i = 0; i < 4; ++i) data[lda * i] = i * 1.0;

  ierr = MatCreateDense(PETSC_COMM_WORLD, 1, PETSC_DECIDE, 2, 4, data, &B);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,"b_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(B, lda);CHKERRQ(ierr);

  /* Test MatMatMult() */
  ierr = MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);
  ierr = MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);

  ierr = MatMatMultEqual(A,B,C,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatMatMult() for C");

  /* Test user-provided mpidense matrix product */
  ierr = MatDuplicate(C,MAT_COPY_VALUES,&C1);CHKERRQ(ierr);
  ierr = MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C1);CHKERRQ(ierr);
  ierr = MatMatMultEqual(A,B,C1,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatMatMult() for C1");

  ierr = MatDestroy(&C1);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  /* Test MatTransposeMatMult() */
  ierr = MatTransposeMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C);CHKERRQ(ierr);

  ierr = MatTransposeMatMultEqual(A,B,C,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatTransposeMatMult()");
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);

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
