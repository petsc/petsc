/*$Id: ex11.c,v 1.12 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = "Tests the use of MatZeroRows() for uniprocessor matrices.\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         i,j,m = 5,n = 5,I,J,ierr;
  Scalar      v,five = 5.0;
  IS          isrow;
  PetscTruth  keepzeroedrows;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C);CHKERRA(ierr);
  ierr = MatSetFromOptions(C);CHKERRA(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  I = j + n*i;
      if (i>0)   {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (i<m-1) {J = I + n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j>0)   {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j<n-1) {J = I + 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,(m*n)/2,0,2,&isrow);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-keep_zeroed_rows",&keepzeroedrows);CHKERRA(ierr);
  if (keepzeroedrows) {
    ierr = MatSetOption(C,MAT_KEEP_ZEROED_ROWS);CHKERRA(ierr);
  }

  ierr = MatZeroRows(C,isrow,&five);CHKERRA(ierr);

  ierr = MatView(C,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = ISDestroy(isrow);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


