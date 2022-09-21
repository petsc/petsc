static char help[] = "Performs adjoint sensitivity analysis for the van der Pol equation.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/* ------------------------------------------------------------------------

   This program solves the van der Pol equation
       y'' - \mu (1-y^2)*y' + y = 0        (1)
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = 0,
   and computes the sensitivities of the final solution w.r.t. initial conditions and parameter \mu with an explicit Runge-Kutta method and its discrete tangent linear model.

   Notes:
   This code demonstrates the TSForward interface to a system of ordinary differential equations (ODEs) in the form of u_t = f(u,t).

   (1) can be turned into a system of first order ODEs
   [ y' ] = [          z          ]
   [ z' ]   [ \mu (1 - y^2) z - y ]

   which then we can write as a vector equation

   [ u_1' ] = [             u_2           ]  (2)
   [ u_2' ]   [ \mu (1 - u_1^2) u_2 - u_1 ]

   which is now in the form of u_t = F(u,t).

   The user provides the right-hand-side function

   [ f(u,t) ] = [ u_2                       ]
                [ \mu (1 - u_1^2) u_2 - u_1 ]

   the Jacobian function

   df   [       0           ;         1        ]
   -- = [                                      ]
   du   [ -2 \mu u_1*u_2 - 1;  \mu (1 - u_1^2) ]

   and the JacobainP (the Jacobian w.r.t. parameter) function

   df      [  0;   0;     0             ]
   ---   = [                            ]
   d\mu    [  0;   0;  (1 - u_1^2) u_2  ]

  ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscmat.h>
typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
  PetscReal tprev;
};

/*
   User-defined routines
*/
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  User               user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(F, &f));
  f[0] = x[1];
  f[1] = user->mu * (1. - x[0] * x[0]) * x[1] - x[0];
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat A, Mat B, void *ctx)
{
  User               user     = (User)ctx;
  PetscReal          mu       = user->mu;
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  J[0][0] = 0;
  J[1][0] = -2. * mu * x[1] * x[0] - 1.;
  J[0][1] = 1.0;
  J[1][1] = mu * (1.0 - x[0] * x[0]);
  PetscCall(MatSetValues(A, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec X, Mat A, void *ctx)
{
  PetscInt           row[] = {0, 1}, col[] = {2};
  PetscScalar        J[2][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  J[0][0] = 0;
  J[1][0] = (1. - x[0] * x[0]) * x[1];
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(MatSetValues(A, 2, row, 1, col, &J[0][0], INSERT_VALUES));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec X, void *ctx)
{
  const PetscScalar *x;
  PetscReal          tfinal, dt, tprev;
  User               user = (User)ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetMaxTime(ts, &tfinal));
  PetscCall(TSGetPrevTime(ts, &tprev));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%.1f] %" PetscInt_FMT " TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n", (double)user->next_output, step, (double)t, (double)dt, (double)PetscRealPart(x[0]), (double)PetscRealPart(x[1])));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "t %.6f (tprev = %.6f) \n", (double)t, (double)tprev));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS             ts;   /* nonlinear solver */
  Vec            x;    /* solution, residual vectors */
  Mat            A;    /* Jacobian matrix */
  Mat            Jacp; /* JacobianP matrix */
  PetscInt       steps;
  PetscReal      ftime   = 0.5;
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar   *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  Mat            sp;

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
  user.mu          = 1;
  user.next_output = 0.0;

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &user.mu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-monitor", &monitor, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatCreateVecs(A, &x, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &Jacp));
  PetscCall(MatSetSizes(Jacp, PETSC_DECIDE, PETSC_DECIDE, 2, 3));
  PetscCall(MatSetFromOptions(Jacp));
  PetscCall(MatSetUp(Jacp));

  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2, 3, NULL, &sp));
  PetscCall(MatZeroEntries(sp));
  PetscCall(MatShift(sp, 1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
  /*   Set RHS Jacobian for the adjoint integration */
  PetscCall(TSSetRHSJacobian(ts, A, A, RHSJacobian, &user));
  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  if (monitor) PetscCall(TSMonitorSet(ts, Monitor, &user, NULL));
  PetscCall(TSForwardSetSensitivities(ts, 3, sp));
  PetscCall(TSSetRHSJacobianP(ts, Jacp, RHSJacobianP, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(x, &x_ptr));

  x_ptr[0] = 2;
  x_ptr[1] = 0.66666654321;
  PetscCall(VecRestoreArray(x, &x_ptr));
  PetscCall(TSSetTimeStep(ts, .001));

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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mu %g, steps %" PetscInt_FMT ", ftime %g\n", (double)user.mu, steps, (double)ftime));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n forward sensitivity: d[y(tf) z(tf)]/d[y0 z0 mu]\n"));
  PetscCall(MatView(sp, PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Jacp));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&sp));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -monitor 0 -ts_adapt_type none

TEST*/
