
static char help[] = "Solves a tridiagonal linear system.\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

  Note:  The corresponding uniprocessor example is ex1.c
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x, b, u;          /* approx solution, RHS, exact solution */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscReal      norm,tol=1000.*PETSC_MACHINE_EPSILON;  /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = 10,col[3],its,rstart,rend,nlocal;
  PetscScalar    one = 1.0,value[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed. For this simple case let PETSc decide how
     many elements of the vector are stored on each processor. The second
     argument to VecSetSizes() below causes PETSc to decide.
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u));

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */

  CHKERRQ(VecGetOwnershipRange(x,&rstart,&rend));
  CHKERRQ(VecGetLocalSize(x,&nlocal));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.

     We pass in nlocal as the "local" size of the matrix to force it
     to have the same parallel layout as the vector created above.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,nlocal,nlocal,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  /*
     Assemble matrix.

     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */

  if (!rstart) {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  }
  if (rend == n) {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
    CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }

  /* Assemble the matrix */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  CHKERRQ(VecSet(u,one));
  CHKERRQ(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create linear solver context
  */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  CHKERRQ(KSPSetOperators(ksp,A,A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCJACOBI));
  CHKERRQ(KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Solve linear system
  */
  CHKERRQ(KSPSolve(ksp,b,x));

  /*
     View solver info; we could instead use the option -ksp_view to
     print this info to the screen at the conclusion of KSPSolve().
  */
  CHKERRQ(KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Check the error
  */
  CHKERRQ(VecAXPY(x,-1.0,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its));
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(VecDestroy(&x)); CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b)); CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      nsize: 3
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3
      nsize: 2
      args: -ksp_monitor_short -ksp_rtol 1e-6 -ksp_type pipefgmres

TEST*/
