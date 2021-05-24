static char help[] = " * Example code testing SeqDense matrices with an LDA (leading dimension of the user-allocated arrray) larger than M.\n";

#include <petscksp.h>

int main(int argc,char **argv)
{
  KSP            solver;
  PC             pc;
  Mat            A,B;
  Vec            X,Y,Z;
  MatScalar      *a;
  PetscScalar    *b,*x,*y,*z;
  PetscReal      nrm;
  PetscErrorCode ierr,size=8,lda=10, i,j;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  /* Create matrix and three vectors: these are all normal */
  ierr = PetscMalloc1(lda*size,&b);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      b[i+j*lda] = rand();
    }
  }
  ierr = MatCreate(MPI_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,size,size,size,size);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A,NULL);CHKERRQ(ierr);

  ierr = MatDenseGetArray(A,&a);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      a[i+j*size] = b[i+j*lda];
    }
  }
  ierr = MatDenseRestoreArray(A,&a);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(MPI_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,size,size,size,size);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(B,b);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(B,lda);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscMalloc1(size,&x);CHKERRQ(ierr);
  for (i=0; i<size; i++) x[i] = 1.0;
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
   * Solve with A and B
   */
  ierr = KSPCreate(MPI_COMM_SELF,&solver);CHKERRQ(ierr);
  ierr = KSPSetType(solver,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(solver,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetOperators(solver,A,A);CHKERRQ(ierr);
  ierr = KSPSolve(solver,X,Y);CHKERRQ(ierr);
  ierr = KSPSetOperators(solver,B,B);CHKERRQ(ierr);
  ierr = KSPSolve(solver,X,Z);CHKERRQ(ierr);
  ierr = VecAXPY(Z,-1.0,Y);CHKERRQ(ierr);
  ierr = VecNorm(Z,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Test1; error norm=%e\n",nrm);CHKERRQ(ierr);

  /* Free spaces */
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);
  ierr = PetscFree(z);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = VecDestroy(&Z);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = KSPDestroy(&solver);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
