/*$Id: ex30.c,v 1.12 1999/05/04 20:33:03 balay Exp bsmith $*/

static char help[] = "Tests ILU factorization and illustrates drawing\n\
of matrix sparsity structure with MatView().  Input parameters are:\n\
  -lf <level> : level of fill for ILU (default is 0)\n\
  -lu : use full LU factorization\n\
  -m <value>, -n <value> : grid dimensions\n\
Note that most users should employ the SLES interface to the\n\
linear solvers instead of using the factorization routines\n\
directly.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C,A; 
  int         i, j, bs, m = 5, n = 5, I, J, ierr, lf = 0,flg1,flg2;
  Scalar      v;
  IS          row,col;
  Viewer      viewer1,viewer2;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg1);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg1);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-lf",&lf,&flg1);CHKERRA(ierr);

  ierr = ViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&viewer1);CHKERRA(ierr);
  ierr = ViewerDrawOpen(PETSC_COMM_SELF,0,0,400,0,400,400,&viewer2);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-mat_bdiag",&flg2);CHKERRA(ierr);
  if (flg2) {
    bs = 1;
    ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg1);CHKERRA(ierr);
    ierr = MatCreateSeqBDiag(PETSC_COMM_SELF,m*n,m*n,0,bs,PETSC_NULL,PETSC_NULL,&C);
          CHKERRA(ierr);
  } else {
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,PETSC_NULL,&C);CHKERRA(ierr);
  }

  /* Create the matrix. (This is five-point stencil with some extra elements) */
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = -1.0;  I = j + n*i;
      J = I - n; if (J>=0)  {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      J = I + n; if (J<m*n) {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      J = I - 1; if (J>=0)  {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      J = I + 1; if (J<m*n) {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = MatGetOrdering(C,MATORDERING_RCM,&row,&col);CHKERRA(ierr);
  printf("original matrix:\n");
  ierr = ViewerPushFormat(VIEWER_STDOUT_SELF,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = ViewerPopFormat(VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatView(C,viewer1);CHKERRA(ierr);

  /* Compute factorization */
  ierr = OptionsHasName(PETSC_NULL,"-lu",&flg1);CHKERRA(ierr);
  if (flg1){ 
    ierr = MatLUFactorSymbolic(C,row,col,1.0,&A);CHKERRA(ierr);
  } else {
    MatILUInfo info;
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    ierr = MatILUFactorSymbolic(C,row,col,&info,&A);CHKERRA(ierr);
  }
  ierr = MatLUFactorNumeric(C,&A);CHKERRA(ierr);

  printf("factored matrix:\n");
  ierr = ViewerPushFormat(VIEWER_STDOUT_SELF,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = ViewerPopFormat(VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatView(A,viewer2);CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = ISDestroy(row);CHKERRA(ierr);
  ierr = ISDestroy(col);CHKERRA(ierr);
  ViewerDestroy(viewer1);
  ViewerDestroy(viewer2);
  PetscFinalize();
  return 0;
}
