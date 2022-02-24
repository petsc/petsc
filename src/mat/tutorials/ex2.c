static char help[] = "testing SeqDense matrices with an LDA (leading dimension of the user-allocated arrray) larger than M.\n";
/*
 * Example code testing SeqDense matrices with an LDA (leading dimension
 * of the user-allocated arrray) larger than M.
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
  PetscErrorCode ierr;
  PetscInt       size=8,size1=6,size2=2, i,j;
  PetscRandom    rnd;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rnd));

  /*
   * Create matrix and three vectors: these are all normal
   */
  CHKERRQ(PetscMalloc1(size*size,&a));
  CHKERRQ(PetscMalloc1(size*size,&b));
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      CHKERRQ(PetscRandomGetValue(rnd,&a[i+j*size]));
      b[i+j*size] = a[i+j*size];
    }
  }
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,size,size,size,size));
  CHKERRQ(MatSetType(A,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(A,a));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscMalloc1(size,&x));
  for (i=0; i<size; i++) {
    x[i] = one;
  }
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size,x,&X));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));

  CHKERRQ(PetscMalloc1(size,&y));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size,y,&Y));
  CHKERRQ(VecAssemblyBegin(Y));
  CHKERRQ(VecAssemblyEnd(Y));

  CHKERRQ(PetscMalloc1(size,&z));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size,z,&Z));
  CHKERRQ(VecAssemblyBegin(Z));
  CHKERRQ(VecAssemblyEnd(Z));

  /*
   * Now create submatrices and subvectors
   */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A11));
  CHKERRQ(MatSetSizes(A11,size1,size1,size1,size1));
  CHKERRQ(MatSetType(A11,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(A11,b));
  CHKERRQ(MatDenseSetLDA(A11,size));
  CHKERRQ(MatAssemblyBegin(A11,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A11,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A12));
  CHKERRQ(MatSetSizes(A12,size1,size2,size1,size2));
  CHKERRQ(MatSetType(A12,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(A12,b+size1*size));
  CHKERRQ(MatDenseSetLDA(A12,size));
  CHKERRQ(MatAssemblyBegin(A12,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A12,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A21));
  CHKERRQ(MatSetSizes(A21,size2,size1,size2,size1));
  CHKERRQ(MatSetType(A21,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(A21,b+size1));
  CHKERRQ(MatDenseSetLDA(A21,size));
  CHKERRQ(MatAssemblyBegin(A21,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A21,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A22));
  CHKERRQ(MatSetSizes(A22,size2,size2,size2,size2));
  CHKERRQ(MatSetType(A22,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(A22,b+size1*size+size1));
  CHKERRQ(MatDenseSetLDA(A22,size));
  CHKERRQ(MatAssemblyBegin(A22,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A22,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size1,x,&X1));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size2,x+size1,&X2));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size1,z,&Z1));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size2,z+size1,&Z2));

  /*
   * Now multiple matrix times input in two ways;
   * compare the results
   */
  CHKERRQ(MatMult(A,X,Y));
  CHKERRQ(MatMult(A11,X1,Z1));
  CHKERRQ(MatMultAdd(A12,X2,Z1,Z1));
  CHKERRQ(MatMult(A22,X2,Z2));
  CHKERRQ(MatMultAdd(A21,X1,Z2,Z2));
  CHKERRQ(VecAXPY(Z,-1.0,Y));
  CHKERRQ(VecNorm(Z,NORM_2,&nrm));
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test1; error norm=%g\n",(double)nrm));
  }

  /*
   * Next test: change both matrices
   */
  CHKERRQ(PetscRandomGetValue(rnd,&v));
  i    = 1;
  j    = size-2;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  j   -= size1;
  CHKERRQ(MatSetValues(A12,1,&i,1,&j,&v,INSERT_VALUES));
  CHKERRQ(PetscRandomGetValue(rnd,&v));
  i    = j = size1+1;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i     =j = 1;
  CHKERRQ(MatSetValues(A22,1,&i,1,&j,&v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(A12,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A12,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(A22,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A22,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatMult(A,X,Y));
  CHKERRQ(MatMult(A11,X1,Z1));
  CHKERRQ(MatMultAdd(A12,X2,Z1,Z1));
  CHKERRQ(MatMult(A22,X2,Z2));
  CHKERRQ(MatMultAdd(A21,X1,Z2,Z2));
  CHKERRQ(VecAXPY(Z,-1.0,Y));
  CHKERRQ(VecNorm(Z,NORM_2,&nrm));
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test2; error norm=%g\n",(double)nrm));
  }

  /*
   * Transpose product
   */
  CHKERRQ(MatMultTranspose(A,X,Y));
  CHKERRQ(MatMultTranspose(A11,X1,Z1));
  CHKERRQ(MatMultTransposeAdd(A21,X2,Z1,Z1));
  CHKERRQ(MatMultTranspose(A22,X2,Z2));
  CHKERRQ(MatMultTransposeAdd(A12,X1,Z2,Z2));
  CHKERRQ(VecAXPY(Z,-1.0,Y));
  CHKERRQ(VecNorm(Z,NORM_2,&nrm));
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test3; error norm=%g\n",(double)nrm));
  }
  CHKERRQ(PetscFree(a));
  CHKERRQ(PetscFree(b));
  CHKERRQ(PetscFree(x));
  CHKERRQ(PetscFree(y));
  CHKERRQ(PetscFree(z));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&A11));
  CHKERRQ(MatDestroy(&A12));
  CHKERRQ(MatDestroy(&A21));
  CHKERRQ(MatDestroy(&A22));

  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(VecDestroy(&Z));

  CHKERRQ(VecDestroy(&X1));
  CHKERRQ(VecDestroy(&X2));
  CHKERRQ(VecDestroy(&Z1));
  CHKERRQ(VecDestroy(&Z2));
  CHKERRQ(PetscRandomDestroy(&rnd));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
