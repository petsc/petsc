static char help[] = "Basic equation for generator stability analysis.\n";

/*F

\begin{eqnarray}
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - \frac{EV}{X} \sin(\theta) -D(\omega - \omega_s)\\
                 \frac{d \theta}{dt} = \omega - \omega_s
\end{eqnarray}

  Ensemble of initial conditions
   ./ex2 -ensemble -ts_monitor_draw_solution_phase -1,-3,3,3      -ts_adapt_dt_max .01  -ts_monitor -ts_type rosw -pc_type lu -ksp_type preonly

  Fault at .1 seconds
   ./ex2           -ts_monitor_draw_solution_phase .42,.95,.6,1.05 -ts_adapt_dt_max .01  -ts_monitor -ts_type rosw -pc_type lu -ksp_type preonly

  Initial conditions same as when fault is ended
   ./ex2 -u 0.496792,1.00932 -ts_monitor_draw_solution_phase .42,.95,.6,1.05  -ts_adapt_dt_max .01  -ts_monitor -ts_type rosw -pc_type lu -ksp_type preonly

F*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscts.h>

typedef struct {
  PetscScalar H, D, omega_s, Pmax, Pm, E, V, X;
  PetscReal   tf, tcl;
} AppCtx;

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, AppCtx *ctx)
{
  PetscScalar       *f, Pmax;
  const PetscScalar *u, *udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else if (t >= ctx->tcl) Pmax = ctx->E / 0.745;
  else Pmax = ctx->Pmax;
  f[0] = udot[0] - ctx->omega_s * (u[1] - 1.0);
  f[1] = 2.0 * ctx->H * udot[1] + Pmax * PetscSinScalar(u[0]) + ctx->D * (u[1] - 1.0) - ctx->Pm;

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat A, Mat B, AppCtx *ctx)
{
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2], Pmax;
  const PetscScalar *u, *udot;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  if ((t > ctx->tf) && (t < ctx->tcl)) Pmax = 0.0; /* A short-circuit on the generator terminal that drives the electrical power output (Pmax*sin(delta)) to 0 */
  else if (t >= ctx->tcl) Pmax = ctx->E / 0.745;
  else Pmax = ctx->Pmax;

  J[0][0] = a;
  J[0][1] = -ctx->omega_s;
  J[1][1] = 2.0 * ctx->H * a + ctx->D;
  J[1][0] = Pmax * PetscCosScalar(u[0]);

  PetscCall(MatSetValues(B, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PostStep(TS ts)
{
  Vec       X;
  PetscReal t;

  PetscFunctionBegin;
  PetscCall(TSGetTime(ts, &t));
  if (t >= .2) {
    PetscCall(TSGetSolution(ts, &X));
    PetscCall(VecView(X, PETSC_VIEWER_STDOUT_WORLD));
    exit(0);
    /* results in initial conditions after fault of -u 0.496792,1.00932 */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution will be stored here */
  Mat          A;  /* Jacobian matrix */
  PetscMPIInt  size;
  PetscInt     n = 2;
  AppCtx       ctx;
  PetscScalar *u;
  PetscReal    du[2]    = {0.0, 0.0};
  PetscBool    ensemble = PETSC_FALSE, flg1, flg2;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A, &U, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Swing equation options", "");
  {
    ctx.omega_s = 2.0 * PETSC_PI * 60.0;
    ctx.H       = 5.0;
    PetscCall(PetscOptionsScalar("-Inertia", "", "", ctx.H, &ctx.H, NULL));
    ctx.D = 5.0;
    PetscCall(PetscOptionsScalar("-D", "", "", ctx.D, &ctx.D, NULL));
    ctx.E    = 1.1378;
    ctx.V    = 1.0;
    ctx.X    = 0.545;
    ctx.Pmax = ctx.E * ctx.V / ctx.X;
    PetscCall(PetscOptionsScalar("-Pmax", "", "", ctx.Pmax, &ctx.Pmax, NULL));
    ctx.Pm = 0.9;
    PetscCall(PetscOptionsScalar("-Pm", "", "", ctx.Pm, &ctx.Pm, NULL));
    ctx.tf  = 1.0;
    ctx.tcl = 1.05;
    PetscCall(PetscOptionsReal("-tf", "Time to start fault", "", ctx.tf, &ctx.tf, NULL));
    PetscCall(PetscOptionsReal("-tcl", "Time to end fault", "", ctx.tcl, &ctx.tcl, NULL));
    PetscCall(PetscOptionsBool("-ensemble", "Run ensemble of different initial conditions", "", ensemble, &ensemble, NULL));
    if (ensemble) {
      ctx.tf  = -1;
      ctx.tcl = -1;
    }

    PetscCall(VecGetArray(U, &u));
    u[0] = PetscAsinScalar(ctx.Pm / ctx.Pmax);
    u[1] = 1.0;
    PetscCall(PetscOptionsRealArray("-u", "Initial solution", "", u, &n, &flg1));
    n = 2;
    PetscCall(PetscOptionsRealArray("-du", "Perturbation in initial solution", "", du, &n, &flg2));
    u[0] += du[0];
    u[1] += du[1];
    PetscCall(VecRestoreArray(U, &u));
    if (flg1 || flg2) {
      ctx.tf  = -1;
      ctx.tcl = -1;
    }
  }
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSROSW));
  PetscCall(TSSetIFunction(ts, NULL, (TSIFunctionFn *)IFunction, &ctx));
  PetscCall(TSSetIJacobian(ts, A, A, (TSIJacobianFn *)IJacobian, &ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts, 35.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, .01));
  PetscCall(TSSetFromOptions(ts));
  /* PetscCall(TSSetPostStep(ts,PostStep));  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (ensemble) {
    for (du[1] = -2.5; du[1] <= .01; du[1] += .1) {
      PetscCall(VecGetArray(U, &u));
      u[0] = PetscAsinScalar(ctx.Pm / ctx.Pmax);
      u[1] = ctx.omega_s;
      u[0] += du[0];
      u[1] += du[1];
      PetscCall(VecRestoreArray(U, &u));
      PetscCall(TSSetTimeStep(ts, .01));
      PetscCall(TSSolve(ts, U));
    }
  } else {
    PetscCall(TSSolve(ts, U));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -nox -ts_dt 10
      output_file: output/empty.out

TEST*/
