#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex8.c,v 1.5 1999/03/19 21:19:59 bsmith Exp bsmith $";
#endif

static char help[] = "Tests automatic allocation of matrix storage space.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         i,j, m = 3, n = 3, I, J, ierr, flg;
  Scalar      v;
  MatInfo     info;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C);CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = -1.0;  I = j + n*i;
      if (i>0)   {J=I-n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if (i<m-1) {J=I+n; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if (j>0)   {J=I-1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      if (j<n-1) {J=I+1; MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  ierr = MatGetInfo(C,MAT_LOCAL,&info); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"matrix nonzeros = %d, allocated nonzeros = %d\n",
    (int)info.nz_used,(int)info.nz_allocated);

  ierr = MatDestroy(C); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
