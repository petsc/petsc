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

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  /* Create matrix and three vectors: these are all normal */
  PetscCall(PetscMalloc1(lda*size,&b));
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      b[i+j*lda] = rand();
    }
  }
  PetscCall(MatCreate(MPI_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,size,size,size,size));
  PetscCall(MatSetType(A,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(A,NULL));

  PetscCall(MatDenseGetArray(A,&a));
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      a[i+j*size] = b[i+j*lda];
    }
  }
  PetscCall(MatDenseRestoreArray(A,&a));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(MPI_COMM_SELF,&B));
  PetscCall(MatSetSizes(B,size,size,size,size));
  PetscCall(MatSetType(B,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(B,b));
  PetscCall(MatDenseSetLDA(B,lda));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscMalloc1(size,&x));
  for (i=0; i<size; i++) x[i] = 1.0;
  PetscCall(VecCreateSeqWithArray(MPI_COMM_SELF,1,size,x,&X));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(PetscMalloc1(size,&y));
  PetscCall(VecCreateSeqWithArray(MPI_COMM_SELF,1,size,y,&Y));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));

  PetscCall(PetscMalloc1(size,&z));
  PetscCall(VecCreateSeqWithArray(MPI_COMM_SELF,1,size,z,&Z));
  PetscCall(VecAssemblyBegin(Z));
  PetscCall(VecAssemblyEnd(Z));

  /*
   * Solve with A and B
   */
  PetscCall(KSPCreate(MPI_COMM_SELF,&solver));
  PetscCall(KSPSetType(solver,KSPPREONLY));
  PetscCall(KSPGetPC(solver,&pc));
  PetscCall(PCSetType(pc,PCLU));
  PetscCall(KSPSetOperators(solver,A,A));
  PetscCall(KSPSolve(solver,X,Y));
  PetscCall(KSPSetOperators(solver,B,B));
  PetscCall(KSPSolve(solver,X,Z));
  PetscCall(VecAXPY(Z,-1.0,Y));
  PetscCall(VecNorm(Z,NORM_2,&nrm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Test1; error norm=%e\n",(double)nrm));

  /* Free spaces */
  PetscCall(PetscFree(b));
  PetscCall(PetscFree(x));
  PetscCall(PetscFree(y));
  PetscCall(PetscFree(z));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(VecDestroy(&Z));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(KSPDestroy(&solver));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
