
static char help[] = "Tests the use of MatZeroRows() for uniprocessor matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 5,n = 5,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v,five = 5.0;
  IS             isrow;
  PetscBool      keepnonzeropattern;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,(m*n)/2,0,2,&isrow);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-keep_nonzero_pattern",&keepnonzeropattern);CHKERRQ(ierr);
  if (keepnonzeropattern) {
    ierr = MatSetOption(C,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = MatZeroRowsIS(C,isrow,five,0,0);CHKERRQ(ierr);

  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}




/*TEST

   test:

   test:
      suffix: 2
      args: -mat_type seqbaij -mat_block_size 5

   test:
      suffix: 3
      args: -keep_nonzero_pattern

   test:
      suffix: 4
      args: -keep_nonzero_pattern -mat_type seqbaij -mat_block_size 5

TEST*/
