/*$Id: ex17.c,v 1.11 2000/05/05 22:18:00 balay Exp bsmith $*/

/* Usage:  mpirun ex2 [-help] [all PETSc options] */

static char help[] = "Solves a linear system in parallel with SLES.\n\
Input parameters include:\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

/*T
   Concepts: Laplacian, 2d
   Processors: n
T*/

/* 
  Include "petscsles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec         x,b,u;  /* approx solution, RHS, exact solution */
  Mat         A;        /* linear system matrix */
  SLES        sles;     /* linear solver context */
  PetscRandom rctx;     /* random number generator context */
  double      norm;     /* norm of solution error */
  int         i,I,Istart,Iend,ierr,m = 5,n = 5,its,*cols;
  Scalar      neg_one = -1.0,*ua;
  PetscTruth  flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"system size: m=%d, n=%d\n",m,n);CHKERRA(ierr);
  if (m < n) SETERRA(1,0,"Supports m >= n only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create parallel vectors.
      - When using VecCreate() and VecSetFromOptions(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime. 
      - When solving a linear system, the vectors and matrices MUST
        be partitioned accordingly.  PETSc automatically generates
        appropriately partitioned matrices and vectors when MatCreate()
        and VecCreate() are used with the same communicator. 
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&u);CHKERRA(ierr);
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&x);CHKERRA(ierr); 
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m,&b);CHKERRA(ierr);
  ierr = VecSetFromOptions(b);CHKERRA(ierr);

  /* 
     Set exact solution with random components.
  */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);CHKERRA(ierr);
  ierr = VecSetRandom(rctx,u);CHKERRA(ierr);
  ierr = VecAssemblyBegin(u);CHKERRA(ierr);
  ierr = VecAssemblyEnd(u);CHKERRA(ierr);

  /* 
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&A);CHKERRA(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRA(ierr);

  /* 
     Set matrix elements in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global rows and columns of matrix entries.
   */
  ierr = VecGetArray(u,&ua);CHKERRA(ierr);
  cols = (int *)PetscMalloc(n*sizeof(int));CHKPTRA(cols);
  for (i=0; i<n; i++) { 
    cols[i] = i;
  }
  for (I=Istart; I<Iend; I++) { 
    ierr = VecSetRandom(rctx,u);CHKERRA(ierr);
    ierr = MatSetValues(A,1,&I,n,cols,ua,INSERT_VALUES);CHKERRA(ierr);
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /*
      Compute right-hand-side vector.
  */
  ierr = MatMult(A,u,b);CHKERRA(ierr);

  /*
     View the exact solution vector if desired
  */
  ierr = OptionsHasName(PETSC_NULL,"-view_exact_sol",&flg);CHKERRA(ierr);
  if (flg) {ierr = VecView(u,VIEWER_STDOUT_WORLD);CHKERRA(ierr);}

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(&neg_one,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);

  /*
     Print convergence information.  PetscPrintf() produces a single 
     print statement from all processes that share a communicator.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %d\n",norm,its);CHKERRA(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rctx);CHKERRA(ierr);
  ierr = PetscFree(cols);CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  PetscFinalize();
  return 0;
}
