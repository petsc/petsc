#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex14.c,v 1.8 1999/05/04 20:33:03 balay Exp bsmith $";
#endif

static char help[] = "Tests MatGetRow() and MatRestoreRow().\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    C; 
  int    i, j, m = 5, n = 5, I, J, ierr, *idx, nz;
  Scalar v, *values;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C);CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  for ( i=0; i<m*n; i++ ) {
    ierr = MatGetRow(C,i,&nz,&idx,&values);CHKERRA(ierr);
#if defined(PETSC_USE_COMPLEX)
    for ( j=0; j<nz; j++ ) PetscPrintf(PETSC_COMM_SELF,"%d %g ",idx[j],PetscReal(values[j]));
#else
    for ( j=0; j<nz; j++ ) PetscPrintf(PETSC_COMM_SELF,"%d %g ",idx[j],values[j]);
#endif
    PetscPrintf(PETSC_COMM_SELF,"\n");
    ierr = MatRestoreRow(C,i,&nz,&idx,&values);CHKERRA(ierr);
  }

  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
