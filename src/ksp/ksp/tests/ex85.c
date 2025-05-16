static char help[] = "Estimate eigenvalues with KSP.\n\n";

/*
    Test example that demonstrates how KSP can estimate eigenvalues.

    Contributed by: Pablo Brubeck <brubeck@protonmail.com>
*/
#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b; /* approx solution, RHS */
  Mat         A;    /* linear system matrix */
  KSP         ksp;  /* linear solver context */
  PC          pc;   /* preconditioner context */
  PetscInt    n = 6, i, j, col[2];
  PetscScalar value[4];
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

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
  value[0] = 2.0;
  value[1] = -1.0;
  value[2] = -1.0;
  value[3] = 2.0;
  for (i = 0; 2 * i < n; i++) {
    col[0] = 2 * i;
    col[1] = 2 * i + 1;
    PetscCall(MatSetValues(A, 2, col, 2, col, value, INSERT_VALUES));
    for (j = 0; j < 4; j++) value[j] *= 3.0;
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*
     Set random right-hand-side vector.
  */
  PetscCall(VecSetRandom(b, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix from which the preconditioner is constructed.
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
  PetscCall(KSPSetTolerances(ksp, 1.e-5, 1.e-5, PETSC_CURRENT, PETSC_CURRENT));

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
  /*
     Solve linear system
  */
  PetscCall(KSPSolve(ksp, b, x));

  /*
     View solver info; we could instead use the option -ksp_view to
     print this info to the screen at the conclusion of KSPSolve().
  */
  PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

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
      suffix: cg_none
      filter: grep -v "variant HERMITIAN"
      args: -ksp_type cg -pc_type none -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: cg_jacobi
      filter: grep -v "variant HERMITIAN"
      args: -ksp_type cg -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: fcg_none
      args: -ksp_type fcg -pc_type none -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: fcg_jacobi
      args: -ksp_type fcg -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: minres_none
      args: -ksp_type minres -pc_type none -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: minres_jacobi
      args: -ksp_type minres -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: gmres_none
      args: -ksp_type gmres -pc_type none -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: gmres_jacobi_left
      args: -ksp_type gmres -ksp_pc_side left -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: gmres_jacobi_right
      args: -ksp_type gmres -ksp_pc_side right -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: fgmres_none
      args: -ksp_type fgmres -pc_type none -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

   test:
      suffix: fgmres_jacobi
      args: -ksp_type fgmres -pc_type jacobi -ksp_view_eigenvalues -ksp_view_singularvalues -ksp_converged_reason

TEST*/
