/*$Id: ex7.c,v 1.11 2000/01/11 21:01:03 bsmith Exp balay $*/

static char help[] = "Tests matrix factorization.  Note that most users should\n\
employ the SLES interface to the linear solvers instead of using the factorization\n\
routines directly.\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C,LU; 
  MatInfo     info;
  int         i,j,m = 3,n = 3,I,J,ierr;
  Scalar      v,mone = -1.0,one = 1.0;
  IS          perm,iperm;
  Vec         x,u,b;
  double      norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&C);CHKERRA(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  I = j + n*i;
      if (i>0)   {J = I - n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (i<m-1) {J = I + n; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j>0)   {J = I - 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      if (j<n-1) {J = I + 1; ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRA(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatGetOrdering(C,MATORDERING_RCM,&perm,&iperm);CHKERRA(ierr);
  ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = ISView(perm,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = MatLUFactorSymbolic(C,perm,iperm,1.0,&LU);CHKERRA(ierr);
  ierr = MatLUFactorNumeric(C,&LU);CHKERRA(ierr);
  ierr = MatView(LU,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,m*n,&u);CHKERRA(ierr);
  ierr = VecSet(&one,u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&x);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);

  ierr = MatMult(C,u,b);CHKERRA(ierr);
  ierr = MatSolve(LU,b,x);CHKERRA(ierr);

  ierr = VecView(b,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = VecAXPY(&mone,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of error %A\n",norm);CHKERRA(ierr);

  ierr = MatGetInfo(C,MAT_LOCAL,&info);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"original matrix nonzeros = %d\n",(int)info.nz_used);CHKERRA(ierr);
  ierr = MatGetInfo(LU,MAT_LOCAL,&info);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"factored matrix nonzeros = %d\n",(int)info.nz_used);CHKERRA(ierr);

  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = ISDestroy(perm);CHKERRA(ierr);
  ierr = ISDestroy(iperm);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatDestroy(LU);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
