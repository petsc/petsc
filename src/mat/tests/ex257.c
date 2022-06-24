static char help[] = "Test MatDenseGetSubMatrix() on a CUDA matrix.\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,B;
  PetscScalar    *b;
  PetscInt       n=4,lda=5,i,k;
  PetscBool      cuda=PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-lda",&lda,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-cuda",&cuda,NULL));
  PetscCheck(lda>=n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"lda %" PetscInt_FMT " < n %" PetscInt_FMT,lda,n);

#if defined(PETSC_HAVE_CUDA)
  if (cuda) PetscCall(MatCreateSeqDenseCUDA(PETSC_COMM_SELF,lda,n,NULL,&A));
  else
#endif
   PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,lda,n,NULL,&A));

  for (k=0;k<3;k++) {
    PetscCall(MatDenseGetSubMatrix(A,0,n,0,n,&B));
    PetscCall(MatDenseGetArray(B,&b));
    for (i=0; i<n; i++) {
      b[i+i*lda] = 2.0*(i+1);
      if (i>0) b[i+(i-1)*lda] = (PetscReal)(k+1);
    }
    PetscCall(MatDenseRestoreArray(B,&b));
    PetscCall(MatDenseRestoreSubMatrix(A,&B));
    PetscCall(MatView(A,NULL));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/ex257_1.out
      diff_args: -j
      test:
         suffix: 1
      test:
         suffix: 1_cuda
         args: -cuda
         requires: cuda
         filter: sed -e "s/seqdensecuda/seqdense/"

TEST*/
