
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
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,NULL,&A);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* take the first diagonal block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,0,1,&isrow);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRQ(ierr);
  B    = *Bsub;
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroySubMatrices(1,&Bsub);CHKERRQ(ierr);

  /* take a strided block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,0,2,&isrow);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRQ(ierr);
  B    = *Bsub;
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroySubMatrices(1,&Bsub);CHKERRQ(ierr);

  /* take the last block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,N-m-1,1,&isrow);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRQ(ierr);
  B    = *Bsub;
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatDestroySubMatrices(1,&Bsub);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
