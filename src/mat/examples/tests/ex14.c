#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex14.c,v 1.2 1997/04/10 00:03:45 bsmith Exp balay $";
#endif

static char help[] = "Tests MatGetRow() and MatRestoreRow().\n";

#include "mat.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat    C; 
  int    i, j, m = 5, n = 5, I, J, ierr, *idx, nz;
  Scalar v, *values;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_SELF,m*n,m*n,&C); CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  for ( i=0; i<m*n; i++ ) {
    ierr = MatGetRow(C,i,&nz,&idx,&values); CHKERRA(ierr);
#if defined(PETSC_COMPLEX)
    for ( j=0; j<nz; j++ ) PetscPrintf(PETSC_COMM_SELF,"%d %g ",idx[j],real(values[j]));
#else
    for ( j=0; j<nz; j++ ) PetscPrintf(PETSC_COMM_SELF,"%d %g ",idx[j],values[j]);
#endif
    PetscPrintf(PETSC_COMM_SELF,"\n");
    ierr = MatRestoreRow(C,i,&nz,&idx,&values); CHKERRA(ierr);
  }

  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
