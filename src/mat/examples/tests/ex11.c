
static char help[] = "Tests the use of MatZeroRows() for uniprocessor matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  PetscInt       i,j,m = 5,n = 5,I,J;
  PetscErrorCode ierr;
  PetscScalar    v,five = 5.0;
  IS             isrow;
  PetscTruth     keepzeroedrows;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  I = j + n*i;
      if (i>0)   {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = I + n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = I + 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,(m*n)/2,0,2,&isrow);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-keep_zeroed_rows",&keepzeroedrows);CHKERRQ(ierr);
  if (keepzeroedrows) {
    ierr = MatSetOption(C,MAT_KEEP_ZEROED_ROWS);CHKERRQ(ierr);
  }

  ierr = MatZeroRows(C,isrow,&five);CHKERRQ(ierr);

  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}


