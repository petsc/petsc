#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.37 1995/12/21 18:33:14 bsmith Exp bsmith $";
#endif

static char help[] = "Solves a tridiagonal linear system with SLES.\n\n";

#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Vec     x, b, u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* SLES context */
  int     ierr, i, n = 10, col[3], its,flg;
  Scalar  none = -1.0, one = 1.0, value[3];
  double  norm;

  PetscInitialize(&argc,&args,0,0,help);
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);

  /* Create vectors */
  ierr = VecCreate(MPI_COMM_WORLD,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&b); CHKERRA(ierr);
  ierr = VecDuplicate(x,&u); CHKERRA(ierr);
  ierr = VecSet(&one,u); CHKERRA(ierr);

  /* Create and assemble matrix */
  ierr = MatCreate(MPI_COMM_WORLD,n,n,&A); CHKERRA(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* Create SLES context; set operators and options; solve linear system */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A, ALLMAT_DIFFERENT_NONZERO_PATTERN);
  CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  ierr = SLESView(sles,STDOUT_VIEWER_WORLD); CHKERRA(ierr);

  /* Check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr  = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12) 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    MPIU_printf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  /* Free work space */
  ierr = VecDestroy(x); CHKERRA(ierr);ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
