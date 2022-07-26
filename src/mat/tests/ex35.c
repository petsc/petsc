
static char help[] = "Tests MatCreateSubMatrices().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,*Bsub;
  PetscInt       i,j,m = 6,n = 6,N = 36,Ii,J;
  PetscScalar    v;
  IS             isrow;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,NULL,&A));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  /* take the first diagonal block */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,m,0,1,&isrow));
  PetscCall(MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub));
  B    = *Bsub;
  PetscCall(ISDestroy(&isrow));
  PetscCall(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatDestroySubMatrices(1,&Bsub));

  /* take a strided block */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,m,0,2,&isrow));
  PetscCall(MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub));
  B    = *Bsub;
  PetscCall(ISDestroy(&isrow));
  PetscCall(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatDestroySubMatrices(1,&Bsub));

  /* take the last block */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,m,N-m-1,1,&isrow));
  PetscCall(MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub));
  B    = *Bsub;
  PetscCall(ISDestroy(&isrow));
  PetscCall(MatView(B,PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatDestroySubMatrices(1,&Bsub));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
