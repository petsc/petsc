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
  PetscBool           equal;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm,&size));

  /*
     Assemble the matrix for the five point stencil, YET AGAIN
  */
  PetscCall(MatCreate(comm,&A1));
  m=2,n=2;
  PetscCall(MatSetSizes(A1,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(A1));
  PetscCall(MatSetUp(A1));
  PetscCall(MatGetOwnershipRange(A1,&istart,&iend));
  for (ii=istart; ii<iend; ii++) {
    v = -1.0; i = ii/n; j = ii - i*n;
    if (i>0)   {J = ii - n; PetscCall(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = ii + n; PetscCall(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = ii - 1; PetscCall(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = ii + 1; PetscCall(MatSetValues(A1,1,&ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; PetscCall(MatSetValues(A1,1,&ii,1,&ii,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A1,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A1,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A1,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDuplicate(A1,MAT_COPY_VALUES,&A2));
  PetscCall(MatDuplicate(A1,MAT_COPY_VALUES,&A3));
  PetscCall(MatDuplicate(A1,MAT_COPY_VALUES,&A4));

  /*create a nest matrix */
  PetscCall(MatCreate(comm,&nest));
  PetscCall(MatSetType(nest,MATNEST));
  mata[0]=A1,mata[1]=A2,mata[2]=A3,mata[3]=A4;
  PetscCall(MatNestSetSubMats(nest,2,NULL,2,NULL,mata));
  PetscCall(MatSetUp(nest));
  PetscCall(MatConvert(nest,MATAIJ,MAT_INITIAL_MATRIX,&aij));
  PetscCall(MatView(aij,PETSC_VIEWER_STDOUT_WORLD));

  /* create a dense matrix */
  PetscCall(MatGetSize(nest,&M,NULL));
  PetscCall(MatGetLocalSize(nest,&m,NULL));
  PetscCall(MatCreateDense(comm,m,PETSC_DECIDE,M,K,NULL,&B));
  PetscCall(MatSetRandom(B,PETSC_NULL));

  /* C = nest*B_dense */
  PetscCall(MatMatMult(nest,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  PetscCall(MatMatMult(nest,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscCall(MatMatMultEqual(nest,B,C,10,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C != nest*B_dense");

  /* Test B = nest*C, reuse C and B with MatProductCreateWithMat() */
  /* C has been obtained from nest*B. Clear internal data structures related to factors to prevent circular references */
  PetscCall(MatProductClear(C));
  PetscCall(MatProductCreateWithMat(nest,C,NULL,B));
  PetscCall(MatProductSetType(B,MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(B));
  PetscCall(MatProductSymbolic(B));
  PetscCall(MatProductNumeric(B));
  PetscCall(MatMatMultEqual(nest,C,B,10,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in B != nest*C_dense");
  PetscCall(MatConvert(nest,MATAIJ,MAT_INPLACE_MATRIX,&nest));
  PetscCall(MatEqual(nest,aij,&equal));
  PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in aij != nest");
  PetscCall(MatDestroy(&nest));

  if (size > 1) { /* Do not know why this test fails for size = 1 */
    PetscCall(MatCreateTranspose(A1,&A5)); /* A1 is symmetric */
    mata[0] = A5;
    PetscCall(MatCreate(comm,&nest));
    PetscCall(MatSetType(nest,MATNEST));
    PetscCall(MatNestSetSubMats(nest,2,NULL,2,NULL,mata));
    PetscCall(MatSetUp(nest));
    PetscCall(MatMatMult(nest,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C1));
    PetscCall(MatMatMult(nest,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C1));

    PetscCall(MatMatMultEqual(nest,B,C1,10,&equal));
    PetscCheck(equal,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in C1 != C");
    PetscCall(MatDestroy(&C1));
    PetscCall(MatDestroy(&A5));
    PetscCall(MatDestroy(&nest));
  }

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&aij));
  PetscCall(MatDestroy(&A1));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&A3));
  PetscCall(MatDestroy(&A4));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

   test:
      suffix: 2

TEST*/
