/*$Id: ex15.c,v 1.20 2000/01/11 21:02:16 bsmith Exp balay $*/

static char help[] = "SLES on an operator with a null space.\n\n";

#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec     x,b,u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* SLES context */
  int     ierr,i,n = 10,col[3],its,i1,i2;
  Scalar  none = -1.0,value[3],avalue;
  double  norm;
  PC      pc;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&b);CHKERRA(ierr);
  ierr = VecDuplicate(x,&u);CHKERRA(ierr);

  /* create a solution that is orthogonal to the constants */
  ierr = VecGetOwnershipRange(u,&i1,&i2);CHKERRA(ierr);
  for (i=i1; i<i2; i++) {
    avalue = i;
    VecSetValues(u,1,&i,&avalue,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(u);CHKERRA(ierr);
  ierr = VecAssemblyEnd(u);CHKERRA(ierr);
  ierr = VecSum(u,&avalue);CHKERRA(ierr);
  avalue = -avalue/(double)n;
  ierr = VecShift(&avalue,u);CHKERRA(ierr);

  /* Create and assemble matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&A);CHKERRA(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1; value[1] = 1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 1.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatMult(A,u,b);CHKERRA(ierr);

  /* Create SLES context; set operators and options; solve linear system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /* Insure that preconditioner has same null space as matrix */
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);

  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
  /* ierr = SLESView(sles,VIEWER_STDOUT_WORLD);CHKERRA(ierr); */

  /* Check error */
  ierr = VecAXPY(&none,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A, Iterations %d\n",norm,its);CHKERRA(ierr);

  /* Free work space */
  ierr = VecDestroy(x);CHKERRA(ierr);ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
