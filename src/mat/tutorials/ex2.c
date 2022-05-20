static char help[] = "testing SeqDense matrices with an LDA (leading dimension of the user-allocated array) larger than M.\n";
/*
 * Example code testing SeqDense matrices with an LDA (leading dimension
 * of the user-allocated array) larger than M.
 * This example tests the functionality of MatSetValues(), MatMult(),
 * and MatMultTranspose().
 */

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,A11,A12,A21,A22;
  Vec            X,X1,X2,Y,Z,Z1,Z2;
  PetscScalar    *a,*b,*x,*y,*z,v,one=1;
  PetscReal      nrm;
  PetscInt       size=8,size1=6,size2=2, i,j;
  PetscRandom    rnd;

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rnd));

  /*
   * Create matrix and three vectors: these are all normal
   */
  PetscCall(PetscMalloc1(size*size,&a));
  PetscCall(PetscMalloc1(size*size,&b));
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      PetscCall(PetscRandomGetValue(rnd,&a[i+j*size]));
      b[i+j*size] = a[i+j*size];
    }
  }
  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,size,size,size,size));
  PetscCall(MatSetType(A,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(A,a));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscMalloc1(size,&x));
  for (i=0; i<size; i++) {
    x[i] = one;
  }
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size,x,&X));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(PetscMalloc1(size,&y));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size,y,&Y));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));

  PetscCall(PetscMalloc1(size,&z));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size,z,&Z));
  PetscCall(VecAssemblyBegin(Z));
  PetscCall(VecAssemblyEnd(Z));

  /*
   * Now create submatrices and subvectors
   */
  PetscCall(MatCreate(PETSC_COMM_SELF,&A11));
  PetscCall(MatSetSizes(A11,size1,size1,size1,size1));
  PetscCall(MatSetType(A11,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(A11,b));
  PetscCall(MatDenseSetLDA(A11,size));
  PetscCall(MatAssemblyBegin(A11,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A11,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_SELF,&A12));
  PetscCall(MatSetSizes(A12,size1,size2,size1,size2));
  PetscCall(MatSetType(A12,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(A12,b+size1*size));
  PetscCall(MatDenseSetLDA(A12,size));
  PetscCall(MatAssemblyBegin(A12,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A12,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_SELF,&A21));
  PetscCall(MatSetSizes(A21,size2,size1,size2,size1));
  PetscCall(MatSetType(A21,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(A21,b+size1));
  PetscCall(MatDenseSetLDA(A21,size));
  PetscCall(MatAssemblyBegin(A21,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A21,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_SELF,&A22));
  PetscCall(MatSetSizes(A22,size2,size2,size2,size2));
  PetscCall(MatSetType(A22,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(A22,b+size1*size+size1));
  PetscCall(MatDenseSetLDA(A22,size));
  PetscCall(MatAssemblyBegin(A22,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A22,MAT_FINAL_ASSEMBLY));

  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size1,x,&X1));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size2,x+size1,&X2));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size1,z,&Z1));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size2,z+size1,&Z2));

  /*
   * Now multiple matrix times input in two ways;
   * compare the results
   */
  PetscCall(MatMult(A,X,Y));
  PetscCall(MatMult(A11,X1,Z1));
  PetscCall(MatMultAdd(A12,X2,Z1,Z1));
  PetscCall(MatMult(A22,X2,Z2));
  PetscCall(MatMultAdd(A21,X1,Z2,Z2));
  PetscCall(VecAXPY(Z,-1.0,Y));
  PetscCall(VecNorm(Z,NORM_2,&nrm));
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test1; error norm=%g\n",(double)nrm));
  }

  /*
   * Next test: change both matrices
   */
  PetscCall(PetscRandomGetValue(rnd,&v));
  i    = 1;
  j    = size-2;
  PetscCall(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  j   -= size1;
  PetscCall(MatSetValues(A12,1,&i,1,&j,&v,INSERT_VALUES));
  PetscCall(PetscRandomGetValue(rnd,&v));
  i    = j = size1+1;
  PetscCall(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i     =j = 1;
  PetscCall(MatSetValues(A22,1,&i,1,&j,&v,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(A12,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A12,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(A22,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A22,MAT_FINAL_ASSEMBLY));

  PetscCall(MatMult(A,X,Y));
  PetscCall(MatMult(A11,X1,Z1));
  PetscCall(MatMultAdd(A12,X2,Z1,Z1));
  PetscCall(MatMult(A22,X2,Z2));
  PetscCall(MatMultAdd(A21,X1,Z2,Z2));
  PetscCall(VecAXPY(Z,-1.0,Y));
  PetscCall(VecNorm(Z,NORM_2,&nrm));
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test2; error norm=%g\n",(double)nrm));
  }

  /*
   * Transpose product
   */
  PetscCall(MatMultTranspose(A,X,Y));
  PetscCall(MatMultTranspose(A11,X1,Z1));
  PetscCall(MatMultTransposeAdd(A21,X2,Z1,Z1));
  PetscCall(MatMultTranspose(A22,X2,Z2));
  PetscCall(MatMultTransposeAdd(A12,X1,Z2,Z2));
  PetscCall(VecAXPY(Z,-1.0,Y));
  PetscCall(VecNorm(Z,NORM_2,&nrm));
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test3; error norm=%g\n",(double)nrm));
  }
  PetscCall(PetscFree(a));
  PetscCall(PetscFree(b));
  PetscCall(PetscFree(x));
  PetscCall(PetscFree(y));
  PetscCall(PetscFree(z));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&A11));
  PetscCall(MatDestroy(&A12));
  PetscCall(MatDestroy(&A21));
  PetscCall(MatDestroy(&A22));

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(VecDestroy(&Z));

  PetscCall(VecDestroy(&X1));
  PetscCall(VecDestroy(&X2));
  PetscCall(VecDestroy(&Z1));
  PetscCall(VecDestroy(&Z2));
  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
