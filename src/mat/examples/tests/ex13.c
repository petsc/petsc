#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex13.c,v 1.6 1999/03/19 21:19:59 bsmith Exp bsmith $";
#endif

static char help[] = 
"Tests copying and ordering uniprocessor row-based sparse matrices.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     C, A;
  int     i, j, m = 5, n = 5, I, J, ierr;
  Scalar  v;
  IS      perm, iperm;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,PETSC_NULL,&C);CHKERRA(ierr);

  /* create the matrix for the five point stencil, YET AGAIN*/
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

  ierr = MatConvert(C,MATSAME,&A); CHKERRA(ierr);

  ierr = MatGetOrdering(A,MATORDERING_ND,&perm,&iperm); CHKERRA(ierr);
  ierr = ISView(perm,VIEWER_STDOUT_SELF); CHKERRA(ierr);
  ierr = ISView(iperm,VIEWER_STDOUT_SELF); CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  ierr = ISDestroy(perm); CHKERRA(ierr);
  ierr = ISDestroy(iperm); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
