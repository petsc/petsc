
static char help[] = "Tests MatCreateSubMatrices().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,*Bsub;
  PetscInt       i,j,m = 6,n = 6,N = 36,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             isrow;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,NULL,&A));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  /* take the first diagonal block */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,m,0,1,&isrow));
  CHKERRQ(MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub));
  B    = *Bsub;
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(MatDestroySubMatrices(1,&Bsub));

  /* take a strided block */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,m,0,2,&isrow));
  CHKERRQ(MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub));
  B    = *Bsub;
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(MatDestroySubMatrices(1,&Bsub));

  /* take the last block */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,m,N-m-1,1,&isrow));
  CHKERRQ(MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub));
  B    = *Bsub;
  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(MatDestroySubMatrices(1,&Bsub));
  CHKERRQ(MatDestroy(&A));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
