static char help[] = "Test some operations of SeqDense matrices with an LDA larger than M.\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B;
  PetscScalar    *a,*b;
  PetscInt       n=4,lda=5,i;

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-lda",&lda,NULL));
  PetscCheck(lda>=n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"lda %" PetscInt_FMT " < n %" PetscInt_FMT,lda,n);

  /*
   * Create two identical matrices (MatDuplicate does not preserve lda)
   */
  PetscCall(PetscCalloc2(lda*n,&a,lda*n,&b));
  for (i=0; i<n; i++) {
    a[i+i*lda] = 1.0+2.0*PETSC_i;
    if (i>0) a[i+(i-1)*lda] = 3.0-0.5*PETSC_i;
    b[i+i*lda] = 1.0+2.0*PETSC_i;
    if (i>0) b[i+(i-1)*lda] = 3.0-0.5*PETSC_i;
  }
  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,n,n,n,n));
  PetscCall(MatSetType(A,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(A,a));
  PetscCall(MatDenseSetLDA(A,lda));

  PetscCall(MatCreate(PETSC_COMM_SELF,&B));
  PetscCall(MatSetSizes(B,n,n,n,n));
  PetscCall(MatSetType(B,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(B,b));
  PetscCall(MatDenseSetLDA(B,lda));

  PetscCall(MatView(A,NULL));
  PetscCall(MatConjugate(A));
  PetscCall(MatView(A,NULL));
  PetscCall(MatRealPart(A));
  PetscCall(MatView(A,NULL));
  PetscCall(MatImaginaryPart(B));
  PetscCall(MatView(B,NULL));

  PetscCall(PetscFree2(a,b));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: complex

   test:

TEST*/
