#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.49 1996/08/22 20:15:12 balay Exp curfman $";
#endif

static char help[] = "Solves a tridiagonal linear system with SLES.\n\n";

/*T
   Concepts: SLES; solving linear equations
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESView();
   Processors: 1
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Vec     x, b, u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* linear solver context */
  double  norm;         /* norm of solution error */
  int     ierr, i, n = 10, col[3], its, flg, size;
  Scalar  none = -1.0, one = 1.0, value[3];

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if (size != 1) SETERRA(1,"This is a uniprocessor example only!");
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);

  /* 
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.
  */
  ierr = MatCreate(MPI_COMM_WORLD,n,n,&A); CHKERRA(ierr);

  /* 
     Assemble matrix
  */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreate(MPI_COMM_WORLD,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&b); CHKERRA(ierr);
  ierr = VecDuplicate(x,&u); CHKERRA(ierr);

  /* 
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* 
     Create linear solver context
  */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);

  /* 
    Set runtime options (e.g., -ksp_type <type> -pc_type <type>)
  */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* 
     Solve linear system
  */
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* 
     View solver options; we could instead use the option -sles_view
  */
  ierr = SLESView(sles,VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  /* 
     Check the error
  */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr  = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12) 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x); CHKERRA(ierr); ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr); ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
