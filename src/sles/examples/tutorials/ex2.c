#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.42 1996/08/16 02:20:45 curfman Exp curfman $";
#endif

static char help[] = "Solves a linear system in parallel with SLES.  To test the\n\
parallel matrix assembly, the matrix is intentionally distributed across the\n\
processors differently from the way it is assembled.\n\n";

/*T
   Concepts: SLES, solving linear equations
   Routines: SLESCreate(), SLESSetOperators(), SLESSetFromOptions()
   Routines: SLESSolve(), SLESView()
   Multiprocessor code
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h - base PETSc routines    mat.h   - matrices
     sys.h   - system routines        ksp.h   - Krylov subspace methods
     is.h    - index sets             pc.h    - preconditioners
     vec.h   - vectors
*/
#include "sles.h"
#include <stdio.h>

int main(int argc,char **args)
{
  Vec     x, b, u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* linear solver context */
  double  norm;
  int     i, j, I, J, Istart, Iend, ierr, m = 8, its, flg;
  Scalar  v, one = 1.0, none = -1.0;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);

  /* Create parallel matrix.  When using MatCreate(), the matrix format
     can be specified at runtime.  Also, the partioning of the matrix is
     determined by PETSc at runtime. */
  ierr = MatCreate(MPI_COMM_WORLD,m*m,m*m,&A); CHKERRA(ierr);

  /* Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);

  /* Assemble matrix for the 2-D, five-point stencil in parallel.
     Each processor needs to insert only elements that it owns
     locally (but any non-local elements will be sent to the
     appropriate processor during matrix assembly). */
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/m; j = I - i*m;  
    if ( i>0 )   {J = I - m; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( i<m-1 ) {J = I + m; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j<m-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    v = 4.0; MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);
  }

  /* Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition,
     by placing code between these two statements */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Create vectors.  Note that we form 1 vector from scratch and
     then duplicate it as needed. */
  ierr = VecCreate(MPI_COMM_WORLD,m*m,&u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr); 
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);

  /* Set exact solution; then compute right-hand-side vector */
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* Create solver context */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);

  /* Set operators (linear system matrix and optional preconditioning matrix) */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /* Process runtime options (e.g., -ksp_type <type> -pc_type <type>) */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* Solve linear system */
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* Check error */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(MPI_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    PetscPrintf(MPI_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* Free work space.  All PETSc objects should be destroyed when they
     are no longer needed. */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
