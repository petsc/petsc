static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers

  Note:  The corresponding parallel example is ex23.c
*/
#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A;       /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PC          pc;      /* preconditioner context */
  PetscReal   norm;    /* norm of solution error */
  PetscInt    i, n = 10, col[3], its;
  PetscMPIInt size;
  PetscScalar value[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_SELF, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Assemble matrix
  */
  value[0] = -1.0;
  value[1] = 2.0;
  value[2] = -1.0;
  for (i = 1; i < n - 1; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
  }
  i      = n - 1;
  col[0] = n - 2;
  col[1] = n - 1;
  PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  i        = 0;
  col[0]   = 0;
  col[1]   = 1;
  value[0] = 2.0;
  value[1] = -1.0;
  PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, b, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Norm of error %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));

  /* check that KSP automatically handles the fact that the the new non-zero values in the matrix are propagated to the KSP solver */
  PetscCall(MatShift(A, 2.0));
  PetscCall(KSPSolve(ksp, b, x));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));

  /* test if prefixes properly propogate to PCMPI objects */
  if (PCMPIServerActive) {
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, "prefix_test_"));
    PetscCall(MatSetOptionsPrefix(A, "prefix_test_"));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(KSPDestroy(&ksp));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2_aijcusparse
      requires: cuda
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda
      args: -ksp_view

   test:
      suffix: 3
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3_aijcusparse
      requires: cuda
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda -ksp_view

   test:
      suffix: aijcusparse
      requires: cuda
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda -ksp_view
      output_file: output/ex1_1_aijcusparse.out

   test:
      requires: defined(PETSC_USE_SINGLE_LIBRARY)
      suffix: mpi_linear_solver_server_1
      nsize: 3
      filter: sed 's?ATOL?RTOL?g' | grep -v HERMITIAN
      # use the MPI Linear Solver Server
      args: -mpi_linear_solver_server -mpi_linear_solver_server_view
      # controls for the use of PCMPI on a particular system
      args: -mpi_linear_solver_server_minimum_count_per_rank 5 -mpi_linear_solver_server_ksp_view -mpi_linear_solver_server_mat_view
      # the usual options for the linear solver (in this case using the server)
      args: -ksp_monitor -ksp_converged_reason -mat_view -ksp_view  -ksp_type cg -pc_type none
      # the options for the prefixed objects
      args: -prefix_test_mpi_linear_solver_server_mat_view -prefix_test_ksp_monitor -prefix_test_mpi_linear_solver_server_minimum_count_per_rank 5

   test:
      requires: !__float128
      suffix: minit
      args: -ksp_monitor -pc_type none -ksp_min_it 8

TEST*/
