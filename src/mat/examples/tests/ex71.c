#ifndef lint
static char vcid[] = "$Id: ex9.c,v 1.4 1995/09/30 19:31:28 bsmith Exp bsmith $";
#endif

static char help[] = "Passes a sparse matrix to Matlab.\n\n";
#include "sles.h"
#include <stdio.h>
#include "petsc.h"

int main(int argc,char **args)
{
  int     ierr,m = 4,n = 5,i,j,I,J;
  Scalar  one = 1.0,v;
  Vec     x;
  Mat     A;
  Viewer  viewer;

  PetscInitialize(&argc,&args,0,0,help);
  OptionsGetInt(0,"-m",&m);
  OptionsGetInt(0,"-n",&n);

  ierr = ViewerMatlabOpen("eagle",-1,&viewer); CHKERRA(ierr);
  ierr = MatCreate(MPI_COMM_WORLD,m*n,m*n,&A); CHKERRA(ierr);

  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatView(A,viewer); CHKERRA(ierr);

  ierr = VecCreateSeq(MPI_COMM_SELF,m,&x); CHKERRA(ierr);
  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecView(x,viewer); CHKERRA(ierr);
  
  PetscSleep(30);
  ierr = PetscObjectDestroy((PetscObject) viewer); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


