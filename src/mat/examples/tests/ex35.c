
static char help[] = "Tests MatGetSubMatrices().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,B,*Bsub;
  PetscInt       i,j,m = 6,n = 6,N = 36,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             isrow;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,PETSC_NULL,&A);CHKERRQ(ierr);
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
  ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRQ(ierr);
  B = *Bsub; 
  ierr = PetscFree(Bsub);CHKERRQ(ierr);
  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);

  /* take a strided block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,0,2,&isrow);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRQ(ierr);
  B = *Bsub; 
  ierr = PetscFree(Bsub);CHKERRQ(ierr);
  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);

  /* take the last block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,N-m-1,1,&isrow);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRQ(ierr);
  B = *Bsub; 
  ierr = PetscFree(Bsub);CHKERRQ(ierr);
  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
 
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

