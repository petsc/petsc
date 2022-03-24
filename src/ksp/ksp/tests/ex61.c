static char help[] = " * Example code testing SeqDense matrices with an LDA (leading dimension of the user-allocated arrray) larger than M.\n";

#include <petscksp.h>

int main(int argc,char **argv)
{
  KSP          solver;
  PC           pc;
  Mat          A,B;
  Vec          X,Y,Z;
  MatScalar   *a;
  PetscScalar *b,*x,*y,*z;
  PetscReal    nrm;
  PetscInt     size = 8,lda=10, i,j;

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  /* Create matrix and three vectors: these are all normal */
  CHKERRQ(PetscMalloc1(lda*size,&b));
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      b[i+j*lda] = rand();
    }
  }
  CHKERRQ(MatCreate(MPI_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,size,size,size,size));
  CHKERRQ(MatSetType(A,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(A,NULL));

  CHKERRQ(MatDenseGetArray(A,&a));
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      a[i+j*size] = b[i+j*lda];
    }
  }
  CHKERRQ(MatDenseRestoreArray(A,&a));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(MPI_COMM_SELF,&B));
  CHKERRQ(MatSetSizes(B,size,size,size,size));
  CHKERRQ(MatSetType(B,MATSEQDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(B,b));
  CHKERRQ(MatDenseSetLDA(B,lda));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscMalloc1(size,&x));
  for (i=0; i<size; i++) x[i] = 1.0;
  CHKERRQ(VecCreateSeqWithArray(MPI_COMM_SELF,1,size,x,&X));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));

  CHKERRQ(PetscMalloc1(size,&y));
  CHKERRQ(VecCreateSeqWithArray(MPI_COMM_SELF,1,size,y,&Y));
  CHKERRQ(VecAssemblyBegin(Y));
  CHKERRQ(VecAssemblyEnd(Y));

  CHKERRQ(PetscMalloc1(size,&z));
  CHKERRQ(VecCreateSeqWithArray(MPI_COMM_SELF,1,size,z,&Z));
  CHKERRQ(VecAssemblyBegin(Z));
  CHKERRQ(VecAssemblyEnd(Z));

  /*
   * Solve with A and B
   */
  CHKERRQ(KSPCreate(MPI_COMM_SELF,&solver));
  CHKERRQ(KSPSetType(solver,KSPPREONLY));
  CHKERRQ(KSPGetPC(solver,&pc));
  CHKERRQ(PCSetType(pc,PCLU));
  CHKERRQ(KSPSetOperators(solver,A,A));
  CHKERRQ(KSPSolve(solver,X,Y));
  CHKERRQ(KSPSetOperators(solver,B,B));
  CHKERRQ(KSPSolve(solver,X,Z));
  CHKERRQ(VecAXPY(Z,-1.0,Y));
  CHKERRQ(VecNorm(Z,NORM_2,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Test1; error norm=%e\n",nrm));

  /* Free spaces */
  CHKERRQ(PetscFree(b));
  CHKERRQ(PetscFree(x));
  CHKERRQ(PetscFree(y));
  CHKERRQ(PetscFree(z));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  CHKERRQ(VecDestroy(&Z));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(KSPDestroy(&solver));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
