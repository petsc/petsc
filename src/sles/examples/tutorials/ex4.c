#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.39 1999/03/19 21:22:11 bsmith Exp bsmith $";
#endif

static char help[] = "Ilustrates using a different preconditioner matrix and\n\
linear system matrix in the SLES solvers.  Note that different storage formats\n\
can be used for the different matrices.\n\n";

/*T
   Concepts: SLES^Different matrices for linear system and preconditioner;
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); VecSetRandom();
   Routines: PetscRandomCreate(); PetscRandomDestroy()
   Processors: n
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

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  SLES        sles;      /* linear solver context */
  Mat         A, B;      /* linear system matrix, preconditioning matrix */
  PetscRandom rctx;      /* random number generator context */
  Vec         x, b, u;   /* approx solution, RHS, exact solution */
  Vec         tmp;       /* work vector */
  Scalar      v,  one = 1.0, scale = 0.0;
  int         i, j, m = 15, n = 17, its, I, J, ierr, Istart, Iend, flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsGetScalar(PETSC_NULL,"-scale",&scale,&flg); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.  Also, create a different
         preconditioner matrix.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create the linear system matrix (A).
      - Here we use a block diagonal matrix format (MATMPIBDIAG) and
        specify only the global size.  The parallel partitioning of
        the matrix will be determined at runtime by PETSc.
  */
  ierr = MatCreateMPIBDiag(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,m*n,
         0,1,PETSC_NULL,PETSC_NULL,&A); CHKERRA(ierr);

  /* 
     Create a different preconditioner matrix (B).  This is usually
     done to form a cheaper (or sparser) preconditioner matrix
     compared to the linear system matrix.
      - Here we use MatCreate(), so that the matrix format and
        parallel partitioning will be determined at runtime.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&B); CHKERRA(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);

  /*
     Set entries within the two matrices
  */
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 ) {
      J=I-n; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(B,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( i<m-1 ) {
      J=I+n; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(B,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( j>0 ) {
      J=I-1; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( j<n-1 ) {
      J=I+1; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    v = 5.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
    v = 3.0; ierr = MatSetValues(B,1,&I,1,&I,&v,INSERT_VALUES); CHKERRA(ierr);
  }

  /*
     Assemble the preconditioner matrix (B), using the 2-step process
       MatAssemblyBegin(), MatAssemblyEnd()
     Note that computations can be done while messages are in
     transition by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -0.5; i = I/n;
    if ( i>1 ) { 
      J=I-(n+1); ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
    if ( i<m-2 ) {
      J=I+n+1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* 
     Assemble the linear system matrix, (A)
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /* 
     Create parallel vectors.
      - When using VecCreate(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime. 
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,&b); CHKERRA(ierr);
  ierr = VecSetFromOptions(b);CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);

  /* 
     Make solution vector be 1 to random noise
  */
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = VecDuplicate(u,&tmp); CHKERRA(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx); CHKERRA(ierr);
  ierr = VecSetRandom(rctx,tmp); CHKERRA(ierr);
  ierr = PetscRandomDestroy(rctx); CHKERRA(ierr);
  ierr = VecAXPY(&scale,tmp,u); CHKERRA(ierr);
  ierr = VecDestroy(tmp); CHKERRA(ierr);

  /*
     Compute right-hand-side vector 
  */
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
    Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles); CHKERRA(ierr);

  /* 
     Set operators. Note that we use different matrices to define the
     linear system and to precondition it.
  */
  ierr = SLESSetOperators(sles,A,B,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /* 
     Set runtime options (e.g., -ksp_type <type> -pc_type <type>)
  */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles); CHKERRA(ierr); ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = MatDestroy(B); CHKERRA(ierr);     ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);     ierr = VecDestroy(b); CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  PetscFinalize();
  return 0;
}
