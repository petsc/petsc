static char help[] = "Solves a one-dimensional steady upwind advection system with KSP.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b, work_vec; /* approx solution, RHS, work vector */
  Mat         A;              /* linear system matrix */
  KSP         ksp;            /* linear solver context */
  PC          pc;             /* preconditioner context */
  PetscInt    i, j, n = 10, col[2];
  PetscScalar work_scalar, value[2];
  PetscRandom r;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &work_vec));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Assemble matrix
  */
  value[0] = -1.0;
  value[1] = 1.0;
  for (i = 1; i < n; i++) {
    col[0] = i - 1;
    col[1] = i;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  }
  i           = 0;
  j           = 0;
  work_scalar = 1;
  PetscCall(MatSetValues(A, 1, &i, 1, &j, &work_scalar, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*
     Create x, b
  */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &r));
  PetscCall(VecSetRandom(x, r));
  PetscCall(VecSetRandom(work_vec, r));
  PetscCall(MatMult(A, work_vec, b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

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
  PetscCall(PCSetType(pc, PCBJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1.e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

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

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&work_vec));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscRandomDestroy(&r));

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

   testset:
      requires: hypre !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_grid_sweeps_down 0 -pc_hypre_boomeramg_grid_sweeps_up 1 -pc_hypre_boomeramg_grid_sweeps_coarse 2 -pc_hypre_boomeramg_max_levels 2 -ksp_rtol 1e-7 -pc_hypre_boomeramg_max_coarse_size 16 -n 33 -ksp_max_it 30 -pc_hypre_boomeramg_relax_type_all Jacobi
      test:
         suffix: hypre
      test:
         suffix: hypre_air
         args: -pc_hypre_boomeramg_restriction_type 1 -pc_hypre_boomeramg_postrelax F -pc_hypre_boomeramg_grid_sweeps_down 1 -pc_hypre_boomeramg_prerelax C

TEST*/
