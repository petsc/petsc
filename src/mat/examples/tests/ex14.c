/*$Id: ex14.c,v 1.12 2000/01/11 21:01:03 bsmith Exp balay $*/

static char help[] = "Tests MatGetRow() and MatRestoreRow().\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    C; 
  int    i,j,m = 5,n = 5,I,J,ierr,*idx,nz;
  Scalar v,*values;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C);CHKERRA(ierr);
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
  ierr = MatView(C,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  for (i=0; i<m*n; i++) {
    ierr = MatGetRow(C,i,&nz,&idx,&values);CHKERRA(ierr);
#if defined(PETSC_USE_COMPLEX)
    for (j=0; j<nz; j++) {ierr = PetscPrintf(PETSC_COMM_SELF,"%d %g ",idx[j],PetscRealPart(values[j]));CHKERRA(ierr);}
#else
    for (j=0; j<nz; j++) {ierr = PetscPrintf(PETSC_COMM_SELF,"%d %g ",idx[j],values[j]);CHKERRA(ierr);}
#endif
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRA(ierr);
    ierr = MatRestoreRow(C,i,&nz,&idx,&values);CHKERRA(ierr);
  }

  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
