#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex17.c,v 1.4 1997/10/19 03:26:38 bsmith Exp bsmith $";
#endif

static char help[] = "Tests the use of MatSolveTrans().\n\n";

#include "mat.h"

int main(int argc,char **args)
{
  Mat     C, A;
  int     i, j, m = 5, n = 5, I, J, ierr, flg;
  Scalar  v, five = 5.0, one = 1.0, mone = -1.0;
  IS      isrow,row,col;
  Vec     x, u, b;
  double  norm;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,PETSC_NULL,&C);CHKERRA(ierr);

  /* create the matrix for the five point stencil, YET AGAIN*/
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
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF,(m*n)/2,0,2,&isrow);CHKERRA(ierr);
  ierr = MatZeroRows(C,isrow,&five); CHKERRA(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,m*n,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&x); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);

  ierr = MatMultTrans(C,u,b); CHKERRA(ierr);

  /* Set default ordering to be Quotient Minimum Degree; also read
     orderings from the options database */
  ierr = MatGetOrdering(C,ORDER_QMD,&row,&col); CHKERRA(ierr);

  ierr = MatLUFactorSymbolic(C,row,col,1.0,&A); CHKERRA(ierr);
  ierr = MatLUFactorNumeric(C,&A); CHKERRA(ierr);
  ierr = MatSolveTrans(A,b,x); CHKERRA(ierr);

  ierr = ISView(row,VIEWER_STDOUT_SELF); CHKERRA(ierr);
  ierr = VecAXPY(&mone,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",norm);

  ierr = ISDestroy(row); CHKERRA(ierr);
  ierr = ISDestroy(col); CHKERRA(ierr);
  ierr = ISDestroy(isrow); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
