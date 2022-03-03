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
  CHKERRMPI(MPI_Comm_size(comm,&size));

  /*
     Assemble the matrix for the five point stencil, YET AGAIN
  */
  CHKERRQ(MatCreate(comm,&A1));
  m=2,n=2;
  CHKERRQ(MatSetSizes(A1,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(A1));
  CHKERRQ(MatSetUp(A1));
  CHKERRQ(MatGetOwnershipRange(A1,&istart,&iend));
  for (ii=istart; ii<iend; ii++) {
    v = -1.0; i = ii/n; j = ii - i*n;
    if (i>0)   {J = ii - n; CHKERRQ(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = ii + n; CHKERRQ(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = ii - 1; CHKERRQ(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = ii + 1; CHKERRQ(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(A1,1,&ii,1,&ii,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A1,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDuplicate(A1,MAT_COPY_VALUES,&A2));
  CHKERRQ(MatDuplicate(A1,MAT_COPY_VALUES,&A3));
  CHKERRQ(MatDuplicate(A1,MAT_COPY_VALUES,&A4));

  /*create a nest matrix */
  CHKERRQ(MatCreate(comm,&nest));
  CHKERRQ(MatSetType(nest,MATNEST));
  mata[0]=A1,mata[1]=A2,mata[2]=A3,mata[3]=A4;
  CHKERRQ(MatNestSetSubMats(nest,2,NULL,2,NULL,mata));
  CHKERRQ(MatSetUp(nest));
  CHKERRQ(MatConvert(nest,MATAIJ,MAT_INITIAL_MATRIX,&aij));
  CHKERRQ(MatView(aij,PETSC_VIEWER_STDOUT_WORLD));

  /* create a dense matrix */
  CHKERRQ(MatGetSize(nest,&M,NULL));
  CHKERRQ(MatGetLocalSize(nest,&m,NULL));
  CHKERRQ(MatCreateDense(comm,m,PETSC_DECIDE,M,K,NULL,&B));
  CHKERRQ(MatSetRandom(B,PETSC_NULL));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* C = nest*B_dense */
  CHKERRQ(MatMatMult(nest,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatMatMult(nest,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatMatMultEqual(nest,B,C,10,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C != nest*B_dense");

  /* Test B = nest*C, reuse C and B with MatProductCreateWithMat() */
  /* C has been obtained from nest*B. Clear internal data structures related to factors to prevent circular references */
  CHKERRQ(MatProductClear(C));
  CHKERRQ(MatProductCreateWithMat(nest,C,NULL,B));
  CHKERRQ(MatProductSetType(B,MATPRODUCT_AB));
  CHKERRQ(MatProductSetFromOptions(B));
  CHKERRQ(MatProductSymbolic(B));
  CHKERRQ(MatProductNumeric(B));
  CHKERRQ(MatMatMultEqual(nest,C,B,10,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in B != nest*C_dense");
  CHKERRQ(MatConvert(nest,MATAIJ,MAT_INPLACE_MATRIX,&nest));
  CHKERRQ(MatEqual(nest,aij,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in aij != nest");
  CHKERRQ(MatDestroy(&nest));

  if (size > 1) { /* Do not know why this test fails for size = 1 */
    CHKERRQ(MatCreateTranspose(A1,&A5)); /* A1 is symmetric */
    mata[0] = A5;
    CHKERRQ(MatCreate(comm,&nest));
    CHKERRQ(MatSetType(nest,MATNEST));
    CHKERRQ(MatNestSetSubMats(nest,2,NULL,2,NULL,mata));
    CHKERRQ(MatSetUp(nest));
    CHKERRQ(MatMatMult(nest,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C1));
    CHKERRQ(MatMatMult(nest,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C1));

    CHKERRQ(MatMatMultEqual(nest,B,C1,10,&equal));
    PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C1 != C");
    CHKERRQ(MatDestroy(&C1));
    CHKERRQ(MatDestroy(&A5));
    CHKERRQ(MatDestroy(&nest));
  }

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&aij));
  CHKERRQ(MatDestroy(&A1));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&A3));
  CHKERRQ(MatDestroy(&A4));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

   test:
      suffix: 2

TEST*/
