#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex37.c,v 1.2 1997/07/09 20:55:45 balay Exp balay $";
#endif

static char help[] = "Tests MatCopy().\n\n"; 

#include "mat.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Mat         C,A; 
  int         i,  n = 10, midx[3], ierr,flg;
  Scalar      v[3];

  PetscInitialize(&argc,&args,(char *)0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  ierr = MatCreate(PETSC_COMM_WORLD,n,n,&C); CHKERRA(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,n,n,&A); CHKERRA(ierr);

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for ( i=1; i<n-1; i++ ){
    midx[2] = i-1; midx[1] = i; midx[0] = i+1;
    ierr = MatSetValues(C,1,&i,3,midx,v,INSERT_VALUES); CHKERRA(ierr);
  }
  i = 0; midx[0] = 0; midx[1] = 1;
  v[0] = 2.0; v[1] = -1.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES); CHKERRA(ierr);
  i = n-1; midx[0] = n-2; midx[1] = n-1;
  v[0] = -1.0; v[1] = 2.; 
  ierr = MatSetValues(C,1,&i,2,midx,v,INSERT_VALUES); CHKERRA(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = MatCopy(C,A); CHKERRA(ierr);

  ierr = MatView(C,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = MatDestroy(C); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

 
