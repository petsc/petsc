#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex24.c,v 1.5 1999/03/19 21:19:59 bsmith Exp balay $";
#endif

static char help[] = "Tests copying an AIJ matrix.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C,A; 
  int         i,j, m = 5, n = 5, I, J, ierr;
  Scalar      v;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,PETSC_NULL,&C);CHKERRA(ierr);
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

  ierr = MatConvert(C,MATSAME,&A);CHKERRA(ierr);

  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
      v = 4.0; MatSetValues(A,1,&I,1,&I,&v,ADD_VALUES);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = MatView(A,VIEWER_STDOUT_SELF);CHKERRA(ierr); 

  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
