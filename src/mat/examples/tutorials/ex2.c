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
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rnd);CHKERRQ(ierr);

  /*
   * Create matrix and three vectors: these are all normal
   */
  ierr = PetscMalloc1(size*size,&a);CHKERRQ(ierr);
  ierr = PetscMalloc1(size*size,&b);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      ierr = PetscRandomGetValue(rnd,&a[i+j*size]);CHKERRQ(ierr);
      b[i+j*size] = a[i+j*size];
    }
  }
  ierr = MatCreate(MPI_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,size,size,size,size);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A,a);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscMalloc1(size,&x);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    x[i] = one;
  }
  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size,x,&X);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);

  ierr = PetscMalloc1(size,&y);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size,y,&Y);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);

  ierr = PetscMalloc1(size,&z);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size,z,&Z);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Z);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Z);CHKERRQ(ierr);

  /*
   * Now create submatrices and subvectors
   */
  ierr = MatCreate(MPI_COMM_SELF,&A11);CHKERRQ(ierr);
  ierr = MatSetSizes(A11,size1,size1,size1,size1);CHKERRQ(ierr);
  ierr = MatSetType(A11,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A11,b);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(A11,size);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A11,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A11,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(MPI_COMM_SELF,&A12);CHKERRQ(ierr);
  ierr = MatSetSizes(A12,size1,size2,size1,size2);CHKERRQ(ierr);
  ierr = MatSetType(A12,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A12,b+size1*size);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(A12,size);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A12,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A12,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(MPI_COMM_SELF,&A21);CHKERRQ(ierr);
  ierr = MatSetSizes(A21,size2,size1,size2,size1);CHKERRQ(ierr);
  ierr = MatSetType(A21,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A21,b+size1);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(A21,size);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A21,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A21,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(MPI_COMM_SELF,&A22);CHKERRQ(ierr);
  ierr = MatSetSizes(A22,size2,size2,size2,size2);CHKERRQ(ierr);
  ierr = MatSetType(A22,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A22,b+size1*size+size1);CHKERRQ(ierr);
  ierr = MatSeqDenseSetLDA(A22,size);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A22,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A22,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size1,x,&X1);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size2,x+size1,&X2);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size1,z,&Z1);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size2,z+size1,&Z2);CHKERRQ(ierr);

  /*
   * Now multiple matrix times input in two ways;
   * compare the results
   */
  ierr = MatMult(A,X,Y);CHKERRQ(ierr);
  ierr = MatMult(A11,X1,Z1);CHKERRQ(ierr);
  ierr = MatMultAdd(A12,X2,Z1,Z1);CHKERRQ(ierr);
  ierr = MatMult(A22,X2,Z2);CHKERRQ(ierr);
  ierr = MatMultAdd(A21,X1,Z2,Z2);CHKERRQ(ierr);
  ierr = VecAXPY(Z,-1.0,Y);CHKERRQ(ierr);
  ierr = VecNorm(Z,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test1; error norm=%g\n",(double)nrm);CHKERRQ(ierr);
  }

  /*
   * Next test: change both matrices
   */
  ierr = PetscRandomGetValue(rnd,&v);CHKERRQ(ierr);
  i    = 1;
  j    = size-2;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  j   -= size1;
  ierr = MatSetValues(A12,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rnd,&v);CHKERRQ(ierr);
  i    = j = size1+1;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  i     =j = 1;
  ierr = MatSetValues(A22,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A12,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A12,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A22,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A22,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatMult(A,X,Y);CHKERRQ(ierr);
  ierr = MatMult(A11,X1,Z1);CHKERRQ(ierr);
  ierr = MatMultAdd(A12,X2,Z1,Z1);CHKERRQ(ierr);
  ierr = MatMult(A22,X2,Z2);CHKERRQ(ierr);
  ierr = MatMultAdd(A21,X1,Z2,Z2);CHKERRQ(ierr);
  ierr = VecAXPY(Z,-1.0,Y);CHKERRQ(ierr);
  ierr = VecNorm(Z,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test2; error norm=%g\n",(double)nrm);CHKERRQ(ierr);
  }

  /*
   * Transpose product
   */
  ierr = MatMultTranspose(A,X,Y);CHKERRQ(ierr);
  ierr = MatMultTranspose(A11,X1,Z1);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(A21,X2,Z1,Z1);CHKERRQ(ierr);
  ierr = MatMultTranspose(A22,X2,Z2);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(A12,X1,Z2,Z2);CHKERRQ(ierr);
  ierr = VecAXPY(Z,-1.0,Y);CHKERRQ(ierr);
  ierr = VecNorm(Z,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm > 100.0*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test3; error norm=%g\n",(double)nrm);CHKERRQ(ierr);
  }
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);
  ierr = PetscFree(z);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&A11);CHKERRQ(ierr);
  ierr = MatDestroy(&A12);CHKERRQ(ierr);
  ierr = MatDestroy(&A21);CHKERRQ(ierr);
  ierr = MatDestroy(&A22);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Z);CHKERRQ(ierr);

  ierr = VecDestroy(&X1);CHKERRQ(ierr);
  ierr = VecDestroy(&X2);CHKERRQ(ierr);
  ierr = VecDestroy(&Z1);CHKERRQ(ierr);
  ierr = VecDestroy(&Z2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
