#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex33.c,v 1.6 1998/12/03 04:01:49 bsmith Exp bsmith $";
#endif

static char help[] = 
"Writes a matrix using the PETSc sparse format. Input arguments are:\n\
   -fout <file> : output file name\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    A;
  Vec    b;
  char   fileout[128];
  int    i, j, m = 6, n = 6, N = 36, ierr, I, J,flg;
  Scalar val, v;
  Viewer view;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateMPIRowbs(PETSC_COMM_WORLD,PETSC_DECIDE,N,6,PETSC_NULL,
         PETSC_NULL,&A); CHKERRA(ierr);
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

  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,N,&b); CHKERRA(ierr);
  for ( i=0; i<N; i++ ) {
    val = i + 1;
    ierr = VecSetValues(b,1,&i,&val,INSERT_VALUES); CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(b); CHKERRA(ierr);
  ierr = VecAssemblyEnd(b); CHKERRA(ierr);

  ierr = OptionsGetString(PETSC_NULL,"-fout",fileout,127,&flg); CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,fileout,BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view); CHKERRA(ierr);
  ierr = VecView(b,view); CHKERRA(ierr);
  ierr = ViewerDestroy(view); CHKERRA(ierr);

  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

