#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex15.c,v 1.8 1999/04/16 16:07:27 bsmith Exp balay $";
#endif

static char help[] = "Tests MatNorm(), MatLUFactor(), MatSolve() and MatSolveAdd().\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         i, j, m = 3, n = 3, I, J, ierr, flg;
  Scalar      v, mone = -1.0, one = 1.0, alpha = 2.0;
  IS          perm, iperm;
  Vec         x, u, b, y;
  double      norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-symmetric",&flg);CHKERRA(ierr);
  if (flg) {  /* Treat matrix as symmetric only if we set this flag */
    ierr = MatSetOption(C,MAT_SYMMETRIC);CHKERRA(ierr);
  }

  /* Create the matrix for the five point stencil, YET AGAIN */
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
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
  ierr = MatGetOrdering(C,MATORDERING_RCM,&perm,&iperm);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ISView(perm,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,m*n,&u);CHKERRA(ierr);
  ierr = VecSet(&one,u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&x);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);
  ierr = VecDuplicate(u,&y);CHKERRA(ierr);
  ierr = MatMult(C,u,b);CHKERRA(ierr);
  ierr = VecCopy(b,y);CHKERRA(ierr);
  ierr = VecScale(&alpha,y);CHKERRA(ierr);

  ierr = MatNorm(C,NORM_FROBENIUS,&norm);CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"Frobenius norm of matrix %g\n",norm);
  ierr = MatNorm(C,NORM_1,&norm);CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"One  norm of matrix %g\n",norm);
  ierr = MatNorm(C,NORM_INFINITY,&norm);CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"Infinity norm of matrix %g\n",norm);

  ierr = MatLUFactor(C,perm,iperm,1.0);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Test MatSolve */
  ierr = MatSolve(C,b,x);CHKERRA(ierr);
  ierr = VecView(b,VIEWER_STDOUT_SELF); CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = VecAXPY(&mone,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  if (norm < 1.e-12) PetscPrintf(PETSC_COMM_SELF,"Norm of error < 1.e-12\n");
  else  PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",norm);

  /* Test MatSolveAdd */
  ierr = MatSolveAdd(C,b,y,x);CHKERRA(ierr);

  ierr = VecAXPY(&mone,y,x);CHKERRA(ierr);
  ierr = VecAXPY(&mone,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);

  if (norm < 1.e-12) PetscPrintf(PETSC_COMM_SELF,"Norm of error < 1.e-12\n");
  else   PetscPrintf(PETSC_COMM_SELF,"Norm of error %g\n",norm);

  ierr = ISDestroy(perm);CHKERRA(ierr);
  ierr = ISDestroy(iperm);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
