#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex20.c,v 1.7 1999/04/16 16:07:27 bsmith Exp balay $";
#endif

static char help[] = "Tests converting a matrix to another format with MatConvert()\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat        C, A; 
  int        i, j, m = 5, n = 4, I, J, ierr, rank;
  PetscTruth set;
  Scalar     v;
  MatType    mtype;

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

 /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C);CHKERRA(ierr);
  for ( i=0; i<m; i++ ) { 
    for ( j=2*rank; j<2*rank+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);
    }
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  
  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,"conv_",&mtype,&set);CHKERRQ(ierr);
  ierr = MatConvert(C,mtype,&A);CHKERRA(ierr);
  ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_IMPL,0);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
