/*$Id: ex33.c,v 1.12 1999/11/05 14:45:44 bsmith Exp bsmith $*/

static char help[] = 
"Writes a matrix using the PETSc sparse format. Input arguments are:\n\
   -fout <file> : output file name\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat        A;
  Vec        b;
  char       fileout[128];
  int        i,j,m = 6,n = 6,N = 36,ierr,I,J;
  PetscTruth flg;
  Scalar     val,v;
  Viewer     view;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = OptionsHasName(PETSC_NULL,"-use_mataij",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,
                           PETSC_DEFAULT,PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,&A);CHKERRA(ierr);
#if defined(PETSC_HAVE_BLOCKSOLVE) && !defined(PETSC_USE_COMPLEX)
  } else {
    ierr = MatCreateMPIRowbs(PETSC_COMM_WORLD,PETSC_DECIDE,N,6,PETSC_NULL,
                             PETSC_NULL,&A);CHKERRA(ierr);
#endif
  }

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  I = j + n*i;
      if (i>0)   {J = I - n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (i<m-1) {J = I + n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j>0)   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j<n-1) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,N,&b);CHKERRA(ierr);
  for (i=0; i<N; i++) {
    val = i + 1;
    ierr = VecSetValues(b,1,&i,&val,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(b);CHKERRA(ierr);
  ierr = VecAssemblyEnd(b);CHKERRA(ierr);

  ierr = OptionsGetString(PETSC_NULL,"-fout",fileout,127,PETSC_NULL);CHKERRA(ierr);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,fileout,BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view);CHKERRA(ierr);
  ierr = VecView(b,view);CHKERRA(ierr);
  ierr = ViewerDestroy(view);CHKERRA(ierr);

  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

