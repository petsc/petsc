#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex71.c,v 1.30 1999/03/19 21:19:59 bsmith Exp bsmith $";
#endif

static char help[] = "Passes a sparse matrix to Matlab.\n\n";
#include "sles.h"
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int     ierr,m = 4,n = 5,i,j,I,J,flg;
  Scalar  one = 1.0,v;
  Vec     x;
  Mat     A;
  Viewer  viewer;

  PetscInitialize(&argc,&args,(char *)0,help);
  OptionsGetInt(PETSC_NULL,"-m",&m,&flg);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  ierr = ViewerSocketOpen(PETSC_COMM_WORLD,"eagle",-1,&viewer); CHKERRA(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&A); CHKERRA(ierr);

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
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatView(A,viewer); CHKERRA(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x); CHKERRA(ierr);
  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecView(x,viewer); CHKERRA(ierr);
  
  PetscSleep(30);
  ierr = PetscObjectDestroy((PetscObject) viewer); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
    


