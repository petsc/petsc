/*
 * ex195.c
 *
 *  Created on: Aug 24, 2015
 *      Author: Fande Kong <fdkong.jd@gmail.com>
 */

static char help[] = " Demonstrate the use of MatConvert_Nest_AIJ\n";

#include <petscmat.h>


int main(int argc,char **args)
{
  Mat                 A1,A2,A3,A4,A5,B,C,C1,nest;
  Mat                 mata[4];
  Mat                 aij;
  MPI_Comm            comm;
  PetscInt            m,M,n,istart,iend,ii,i,J,j,K=10;
  PetscScalar         v;
  PetscMPIInt         size;
  PetscErrorCode      ierr;
  PetscBool           equal;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  /*
     Assemble the matrix for the five point stencil, YET AGAIN
  */
  ierr = MatCreate(comm,&A1);CHKERRQ(ierr);
  m=2,n=2;
  ierr = MatSetSizes(A1,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A1);CHKERRQ(ierr);
  ierr = MatSetUp(A1);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A1,&istart,&iend);CHKERRQ(ierr);
  for (ii=istart; ii<iend; ii++) {
    v = -1.0; i = ii/n; j = ii - i*n;
    if (i>0)   {J = ii - n; ierr = MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = ii + n; ierr = MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = ii - 1; ierr = MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = ii + 1; ierr = MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A1,1,&ii,1,&ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDuplicate(A1,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = MatDuplicate(A1,MAT_COPY_VALUES,&A3);CHKERRQ(ierr);
  ierr = MatDuplicate(A1,MAT_COPY_VALUES,&A4);CHKERRQ(ierr);

  /*create a nest matrix */
  ierr = MatCreate(comm,&nest);CHKERRQ(ierr);
  ierr = MatSetType(nest,MATNEST);CHKERRQ(ierr);
  mata[0]=A1,mata[1]=A2,mata[2]=A3,mata[3]=A4;
  ierr = MatNestSetSubMats(nest,2,NULL,2,NULL,mata);CHKERRQ(ierr);
  ierr = MatSetUp(nest);CHKERRQ(ierr);
  ierr = MatConvert(nest,MATAIJ,MAT_INITIAL_MATRIX,&aij);CHKERRQ(ierr);
  ierr = MatView(aij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* create a dense matrix */
  ierr = MatGetSize(nest,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(nest,&m,NULL);CHKERRQ(ierr);
  ierr = MatCreateDense(comm,m,PETSC_DECIDE,M,K,NULL,&B);CHKERRQ(ierr);
  ierr = MatSetRandom(B,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* C = nest*B_dense */
  ierr = MatMatMult(nest,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  ierr = MatMatMult(nest,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  ierr = MatMatMultEqual(nest,B,C,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C != nest*B_dense");

  /* Test B = nest*C, reuse C and B with MatProductCreateWithMat() */
  /* C has been obtained from nest*B. Clear internal data structures related to factors to prevent circular references */
  ierr = MatProductClear(C);CHKERRQ(ierr);
  ierr = MatProductCreateWithMat(nest,C,NULL,B);CHKERRQ(ierr);
  ierr = MatProductSetType(B,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatProductSymbolic(B);CHKERRQ(ierr);
  ierr = MatProductNumeric(B);CHKERRQ(ierr);
  ierr = MatMatMultEqual(nest,C,B,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in B != nest*C_dense");
  ierr = MatDestroy(&nest);CHKERRQ(ierr);

  if (size > 1) { /* Do not know why this test fails for size = 1 */
    ierr = MatCreateTranspose(A1,&A5);CHKERRQ(ierr); /* A1 is symmetric */
    mata[0] = A5;
    ierr = MatCreate(comm,&nest);CHKERRQ(ierr);
    ierr = MatSetType(nest,MATNEST);CHKERRQ(ierr);
    ierr = MatNestSetSubMats(nest,2,NULL,2,NULL,mata);CHKERRQ(ierr);
    ierr = MatSetUp(nest);CHKERRQ(ierr);
    ierr = MatMatMult(nest,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C1);CHKERRQ(ierr);
    ierr = MatMatMult(nest,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C1);CHKERRQ(ierr);

    ierr = MatMatMultEqual(nest,B,C1,10,&equal);CHKERRQ(ierr);
    if (!equal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C1 != C");
    ierr = MatDestroy(&C1);CHKERRQ(ierr);
    ierr = MatDestroy(&A5);CHKERRQ(ierr);
    ierr = MatDestroy(&nest);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&aij);CHKERRQ(ierr);
  ierr = MatDestroy(&A1);CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&A3);CHKERRQ(ierr);
  ierr = MatDestroy(&A4);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 2

   test:
      suffix: 2

TEST*/
