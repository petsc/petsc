#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.38 1996/01/12 22:08:38 bsmith Exp $";
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
  OptionsGetInt(PETSC_NULL,"-n",&n,&flg);
  OptionsGetInt(PETSC_NULL,"-kind",&kind,&flg);

  /* Create vectors */
  ierr = VecCreate(MPI_COMM_WORLD,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&b); CHKERRA(ierr);
  ierr = VecDuplicate(x,&u); CHKERRA(ierr);
  ierr = SYRandomCreate(RANDOM_DEFAULT,&rctx); CHKERRA(ierr);
  ierr = VecSetRandom(rctx,u); CHKERRA(ierr);

  /* Create and assemble matrix */
  ierr = MatCreate(MPI_COMM_WORLD,n,n,&A); CHKERRA(ierr);
  ierr = FormTestMatrix(A,n,kind); CHKERRQ(ierr);
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

int FormTestMatrix(Mat A,int n,int kind)
{
  Scalar value[4];
  int    i, ierr, col[4];

  if (kind == 0) {
    value[0] = 1.0; value[1] = 4.0; value[2] = -2.0;
    for (i=1; i<n-1; i++ ) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
    }
    i = n - 1; col[0] = n - 2; col[1] = n - 1;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
    i = 0; col[0] = 0; col[1] = 1; value[0] = 4.0; value[1] = -2.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
  } 
  else if (kind == 1) {
    value[0] = 1.0; value[1] = 0.0; value[2] = 2.0; value[3] = 1.0;
    for (i=2; i<n-2; i++ ) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
    }
    i = n - 2; col[0] = n - 3; col[1] = n - 2; col[1] = n - 1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
    i = n - 1; col[0] = n - 2; col[1] = n - 1;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
    i = 0; col[0] = 0; col[1] = 1;
    ierr = MatSetValues(A,1,&i,2,col,&value[2],INSERT_VALUES); CHKERRA(ierr);
    i = 1; col[0] = 0; col[1] = 1; col[2] = 2;
    ierr = MatSetValues(A,1,&i,3,col,&value[1],INSERT_VALUES); CHKERRA(ierr);
  } 
  else SETERRQ(1,"FormTestMatrix: this kind of test matrix not supported");

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  return 0;
}
