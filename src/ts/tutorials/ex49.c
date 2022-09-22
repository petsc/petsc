
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/* ------------------------------------------------------------------------

   This program solves the van der Pol DAE ODE equivalent
       y' = z                 (1)
       z' = mu[(1-y^2)z-y]
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = -6.6e-01,
   and
       mu = 10^6.
   This is a nonlinear equation.

   This is a copy and modification of ex20.c to exactly match a test
   problem that comes with the Radau5 integrator package.

  ------------------------------------------------------------------------- */

#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
};

static PetscErrorCode IFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  User               user = (User)ctx;
  const PetscScalar *x, *xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArray(F, &f));
  f[0] = xdot[0] - x[1];
  f[1] = xdot[1] - user->mu * ((1.0 - x[0] * x[0]) * x[1] - x[0]);
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat A, Mat B, void *ctx)
{
  User               user     = (User)ctx;
  PetscInt           rowcol[] = {0, 1};
  const PetscScalar *x;
  PetscScalar        J[2][2];

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  J[0][0] = a;
  J[0][1] = -1.0;
  J[1][0] = user->mu * (1.0 + 2.0 * x[0] * x[1]);
  J[1][1] = a - user->mu * (1.0 - x[0] * x[0]);
  PetscCall(MatSetValues(B, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &x));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS             ts; /* nonlinear solver */
  Vec            x;  /* solution, residual vectors */
  Mat            A;  /* Jacobian matrix */
  PetscInt       steps;
  PetscReal      ftime = 2;
  PetscScalar   *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e6;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Physical parameters", NULL);
  PetscCall(PetscOptionsReal("-mu", "Stiffness parameter", "<1.0e6>", user.mu, &user.mu, NULL));
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A, &x, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, &user));
  PetscCall(TSSetIJacobian(ts, A, A, IJacobian, &user));

  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTolerances(ts, 1.e-4, NULL, 1.e-4, NULL));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(x, &x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -6.6e-01;
  PetscCall(VecRestoreArray(x, &x_ptr));
  PetscCall(TSSetTimeStep(ts, .000001));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, x));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "steps %" PetscInt_FMT ", ftime %g\n", steps, (double)ftime));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
  return (0);
}

/*TEST

    build:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES) radau5

    test:
      args: -ts_monitor_solution -ts_type radau5

TEST*/
