#ifndef lint
static char vcid[] = "$Id: ex17.c,v 1.2 1996/01/23 00:20:56 curfman Exp curfman $";
#endif

static char help[] = "Solves a linear system with SLES.  This problem is\n\
intended to test the complex numbers version of various solvers.\n\n";

#include "sles.h"
#include <stdio.h>

int FormTestMatrix(Mat,int,int);

int main(int argc,char **args)
{
  Vec      x, b, u;      /* approx solution, RHS, exact solution */
  Mat      A;            /* linear system matrix */
  SLES     sles;         /* SLES context */
  int      ierr, n = 10, kind=0, its, flg;
  Scalar   none = -1.0;
  double   norm;
  SYRandom rctx;

  PetscInitialize(&argc,&args,0,0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-kind",&kind,&flg); CHKERRA(ierr);

  /* Create vectors */
  ierr = VecCreate(MPI_COMM_WORLD,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&b); CHKERRA(ierr);
  ierr = VecDuplicate(x,&u); CHKERRA(ierr);
  ierr = SYRandomCreate(MPI_COMM_WORLD,RANDOM_DEFAULT,&rctx); CHKERRA(ierr);
  ierr = VecSetRandom(rctx,u); CHKERRA(ierr);

  /* Create and assemble matrix */
  ierr = MatCreate(MPI_COMM_WORLD,n,n,&A); CHKERRA(ierr);
  ierr = FormTestMatrix(A,n,kind); CHKERRQ(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-printout",&flg); CHKERRA(ierr);
  if (flg) {
    ierr = MatView(A,STDOUT_VIEWER_WORLD); CHKERRA(ierr);
    ierr = VecView(u,STDOUT_VIEWER_WORLD); CHKERRA(ierr);
    ierr = VecView(b,STDOUT_VIEWER_WORLD); CHKERRA(ierr);
  }

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
  ierr = VecDestroy(x); CHKERRA(ierr); ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr); ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = SYRandomDestroy(rctx); CHKERRQ(ierr);
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

int FormTestMatrix(Mat A,int n,int kind)
{
#if !defined(PETSC_COMPLEX)
  SETERRQ(1,"FormTestMatrix: These problems require complex numbers.");
#else

  Scalar val[5];
  int    i, ierr, col[5];

  if (kind == 0) {
    val[0] = 1.0; val[1] = 4.0; val[2] = -2.0;
    for (i=1; i<n-1; i++ ) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,val,INSERT_VALUES); CHKERRQ(ierr);
    }
    i = n-1; col[0] = n-2; col[1] = n-1;
    ierr = MatSetValues(A,1,&i,2,col,val,INSERT_VALUES); CHKERRQ(ierr);
    i = 0; col[0] = 0; col[1] = 1; val[0] = 4.0; val[1] = -2.0;
    ierr = MatSetValues(A,1,&i,2,col,val,INSERT_VALUES); CHKERRQ(ierr);
  } 
  else if (kind == 1) {
    val[0] = 1.0; val[1] = 0.0; val[2] = 2.0; val[3] = 1.0;
    for (i=2; i<n-1; i++ ) {
      col[0] = i-2; col[1] = i-1; col[2] = i; col[3] = i+1;
      ierr = MatSetValues(A,1,&i,4,col,val,INSERT_VALUES); CHKERRQ(ierr);
    }
    i = n-1; col[0] = n-3; col[1] = n-2; col[2] = n-1;
    ierr = MatSetValues(A,1,&i,3,col,val,INSERT_VALUES); CHKERRQ(ierr);
    i = 1; col[0] = 0; col[1] = 1; col[2] = 2;
    ierr = MatSetValues(A,1,&i,3,col,&val[1],INSERT_VALUES); CHKERRQ(ierr);
    i = 0;
    ierr = MatSetValues(A,1,&i,2,col,&val[2],INSERT_VALUES); CHKERRQ(ierr);
  } 
  else if (kind == 2) {
    complex tmp(0.0,2.0);
    val[0] = tmp;
    val[1] = 4.0; val[2] = 0.0; val[3] = 1.0; val[4] = 0.7;
    for (i=1; i<n-3; i++ ) {
      col[0] = i-1; col[1] = i; col[2] = i+1; col[3] = i+2; col[4] = i+3;
      ierr = MatSetValues(A,1,&i,5,col,val,INSERT_VALUES); CHKERRQ(ierr);
    }
    i = n-3; col[0] = n-4; col[1] = n-3; col[2] = n-2; col[3] = n-1;
    ierr = MatSetValues(A,1,&i,4,col,val,INSERT_VALUES); CHKERRQ(ierr);
    i = n-2; col[0] = n-3; col[1] = n-2; col[2] = n-1;
    ierr = MatSetValues(A,1,&i,3,col,val,INSERT_VALUES); CHKERRQ(ierr);
    i = n-1; col[0] = n-2; col[1] = n-1;
    ierr = MatSetValues(A,1,&i,2,col,val,INSERT_VALUES); CHKERRQ(ierr);
    i = 0; col[0] = 0; col[1] = 1; col[2] = 2; col[3] = 3;
    ierr = MatSetValues(A,1,&i,3,col,&val[1],INSERT_VALUES); CHKERRQ(ierr);
  } 
  else SETERRQ(1,"FormTestMatrix: this kind of test matrix not supported");

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
#endif

  return 0;
}
