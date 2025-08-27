static char help[] = "Newton's method to solve a two-variable system that comes from the Rosenbrock function.\n\n";

/*
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscsnes.h>

extern PetscErrorCode FormJacobian1(SNES, Vec, Mat, Mat, void *);
extern PetscErrorCode FormFunction1(SNES, Vec, Vec, void *);

int main(int argc, char **argv)
{
  SNES                snes; /* nonlinear solver context */
  Vec                 x;    /* solution vector */
  Mat                 J;    /* Jacobian matrix */
  PetscInt            its;
  PetscScalar        *xx;
  SNESConvergedReason reason;
  PetscBool           test_ghost = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_ghost", &test_ghost, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors for solution and nonlinear function
  */
  if (test_ghost) {
    PetscInt gIdx[] = {0, 1};
    PetscInt nghost = 2;

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-nghost", &nghost, NULL));
    PetscCall(VecCreateGhost(PETSC_COMM_WORLD, 2, PETSC_DECIDE, PetscMin(nghost, 2), gIdx, &x));
  } else {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, 2, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(x));
  }

  /*
     Create Jacobian matrix data structure
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, 2, 2, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  /*
     Set function evaluation routine and vector.
  */
  PetscCall(SNESSetFunction(snes, NULL, FormFunction1, &test_ghost));

  /*
     Set Jacobian matrix data structure and Jacobian evaluation routine
  */
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian1, &test_ghost));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(x, &xx));
  xx[0] = -1.2;
  xx[1] = 1.0;
  PetscCall(VecRestoreArray(x, &xx));

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(VecViewFromOptions(x, NULL, "-sol_view"));
  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  /*
     Some systems computes a residual that is identically zero, thus converging
     due to FNORM_ABS, others converge due to FNORM_RELATIVE.  Here, we only
     report the reason if the iteration did not converge so that the tests are
     reproducible.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s number of SNES iterations = %" PetscInt_FMT "\n", reason > 0 ? "CONVERGED" : SNESConvergedReasons[reason], its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode VecCheckGhosted(Vec X, PetscBool test_rev)
{
  PetscFunctionBeginUser;
  PetscCall(VecGhostUpdateBegin(X, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGhostUpdateEnd(X, INSERT_VALUES, SCATTER_FORWARD));
  if (test_rev) {
    PetscCall(VecGhostUpdateBegin(X, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecGhostUpdateEnd(X, INSERT_VALUES, SCATTER_REVERSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormFunction1(SNES snes, Vec x, Vec f, void *ctx)
{
  PetscScalar       *ff;
  const PetscScalar *xx;

  PetscFunctionBeginUser;
  if (*(PetscBool *)ctx) {
    PetscCall(VecCheckGhosted(x, PETSC_FALSE));
    PetscCall(VecCheckGhosted(f, PETSC_TRUE));
  }

  /*
    Get pointers to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
    the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
    the array.
  */
  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(f, &ff));

  /* Compute function */
  ff[0] = -2.0 + 2.0 * xx[0] + 400.0 * xx[0] * xx[0] * xx[0] - 400.0 * xx[0] * xx[1];
  ff[1] = -200.0 * xx[0] * xx[0] + 200.0 * xx[1];

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(f, &ff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian1(SNES snes, Vec x, Mat jac, Mat B, void *ctx)
{
  const PetscScalar *xx;
  PetscScalar        A[4];
  PetscInt           idx[2];
  PetscMPIInt        rank;

  PetscFunctionBeginUser;
  if (*(PetscBool *)ctx) PetscCall(VecCheckGhosted(x, PETSC_FALSE));
  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x, &xx));

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0] = 2.0 + 1200.0 * xx[0] * xx[0] - 400.0 * xx[1];
  A[1] = -400.0 * xx[0];
  A[2] = -400.0 * xx[0];
  A[3] = 200;

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)x), &rank));
  idx[0] = 2 * rank;
  idx[1] = 2 * rank + 1;

  PetscCall(MatSetValues(B, 2, idx, 2, idx, A, INSERT_VALUES));

  /*
     Restore vector
  */
  PetscCall(VecRestoreArrayRead(x, &xx));

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (jac != B) {
    PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      suffix: 1
      args: -snes_monitor_short -snes_max_it 1000 -sol_view
      requires: !single

   test:
      suffix: 2
      args: -snes_monitor_short -snes_max_it 1000 -snes_type newtontrdc -snes_trdc_use_cauchy false -sol_view
      requires: !single

   test:
      suffix: ghosts
      nsize: {{1 2}}
      args: -snes_max_it 4 -snes_type {{newtontr newtonls}} -nghost {{0 1 2}} -test_ghost
      requires: !single

TEST*/
