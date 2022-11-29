
static char help[] = "Adjoint and tangent linear sensitivity analysis of the basic equation for generator stability analysis.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

F*/

/*
  This code demonstrate the sensitivity analysis interface to a system of ordinary differential equations with discontinuities.
  It computes the sensitivities of an integral cost function
  \int c*max(0,\theta(t)-u_s)^beta dt
  w.r.t. initial conditions and the parameter P_m.
  Backward Euler method is used for time integration.
  The discontinuities are detected with TSEvent.
 */

#include <petscts.h>
#include "ex3.h"

int main(int argc, char **argv)
{
  TS           ts, quadts; /* ODE integrator */
  Vec          U;          /* solution will be stored here */
  PetscMPIInt  size;
  PetscInt     n = 2;
  AppCtx       ctx;
  PetscScalar *u;
  PetscReal    du[2]    = {0.0, 0.0};
  PetscBool    ensemble = PETSC_FALSE, flg1, flg2;
  PetscReal    ftime;
  PetscInt     steps;
  PetscScalar *x_ptr, *y_ptr, *s_ptr;
  Vec          lambda[1], q, mu[1];
  PetscInt     direction[2];
  PetscBool    terminate[2];
  Mat          qgrad;
  Mat          sp; /* Forward sensitivity matrix */
  SAMethod     sa;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &ctx.Jac));
  PetscCall(MatSetSizes(ctx.Jac, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(ctx.Jac, MATDENSE));
  PetscCall(MatSetFromOptions(ctx.Jac));
  PetscCall(MatSetUp(ctx.Jac));
  PetscCall(MatCreateVecs(ctx.Jac, &U, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &ctx.Jacp));
  PetscCall(MatSetSizes(ctx.Jacp, PETSC_DECIDE, PETSC_DECIDE, 2, 1));
  PetscCall(MatSetFromOptions(ctx.Jacp));
  PetscCall(MatSetUp(ctx.Jacp));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, &ctx.DRDP));
  PetscCall(MatSetUp(ctx.DRDP));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, &ctx.DRDU));
  PetscCall(MatSetUp(ctx.DRDU));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Swing equation options", "");
  {
    ctx.beta    = 2;
    ctx.c       = 10000.0;
    ctx.u_s     = 1.0;
    ctx.omega_s = 1.0;
    ctx.omega_b = 120.0 * PETSC_PI;
    ctx.H       = 5.0;
    PetscCall(PetscOptionsScalar("-Inertia", "", "", ctx.H, &ctx.H, NULL));
    ctx.D = 5.0;
    PetscCall(PetscOptionsScalar("-D", "", "", ctx.D, &ctx.D, NULL));
    ctx.E        = 1.1378;
    ctx.V        = 1.0;
    ctx.X        = 0.545;
    ctx.Pmax     = ctx.E * ctx.V / ctx.X;
    ctx.Pmax_ini = ctx.Pmax;
    PetscCall(PetscOptionsScalar("-Pmax", "", "", ctx.Pmax, &ctx.Pmax, NULL));
    ctx.Pm = 1.1;
    PetscCall(PetscOptionsScalar("-Pm", "", "", ctx.Pm, &ctx.Pm, NULL));
    ctx.tf  = 0.1;
    ctx.tcl = 0.2;
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
    sa = SA_ADJ;
    PetscCall(PetscOptionsEnum("-sa_method", "Sensitivity analysis method (adj or tlm)", "", SAMethods, (PetscEnum)sa, (PetscEnum *)&sa, NULL));
  }
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetRHSFunction(ts, NULL, (TSRHSFunction)RHSFunction, &ctx));
  PetscCall(TSSetRHSJacobian(ts, ctx.Jac, ctx.Jac, (TSRHSJacobian)RHSJacobian, &ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts, U));

  /*   Set RHS JacobianP */
  PetscCall(TSSetRHSJacobianP(ts, ctx.Jacp, RHSJacobianP, &ctx));

  PetscCall(TSCreateQuadratureTS(ts, PETSC_FALSE, &quadts));
  PetscCall(TSSetRHSFunction(quadts, NULL, (TSRHSFunction)CostIntegrand, &ctx));
  PetscCall(TSSetRHSJacobian(quadts, ctx.DRDU, ctx.DRDU, (TSRHSJacobian)DRDUJacobianTranspose, &ctx));
  PetscCall(TSSetRHSJacobianP(quadts, ctx.DRDP, DRDPJacobianTranspose, &ctx));
  if (sa == SA_ADJ) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Save trajectory of solution so that TSAdjointSolve() may be used
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(TSSetSaveTrajectory(ts));
    PetscCall(MatCreateVecs(ctx.Jac, &lambda[0], NULL));
    PetscCall(MatCreateVecs(ctx.Jacp, &mu[0], NULL));
    PetscCall(TSSetCostGradients(ts, 1, lambda, mu));
  }

  if (sa == SA_TLM) {
    PetscScalar val[2];
    PetscInt    row[] = {0, 1}, col[] = {0};

    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, &qgrad));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, &sp));
    PetscCall(TSForwardSetSensitivities(ts, 1, sp));
    PetscCall(TSForwardSetSensitivities(quadts, 1, qgrad));
    val[0] = 1. / PetscSqrtScalar(1. - (ctx.Pm / ctx.Pmax) * (ctx.Pm / ctx.Pmax)) / ctx.Pmax;
    val[1] = 0.0;
    PetscCall(MatSetValues(sp, 2, row, 1, col, val, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(sp, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(sp, MAT_FINAL_ASSEMBLY));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, 0.03125));
  PetscCall(TSSetFromOptions(ts));

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;

  PetscCall(TSSetEventHandler(ts, 2, direction, terminate, EventFunction, PostEventFunction, (void *)&ctx));

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
      PetscCall(TSSetTimeStep(ts, 0.03125));
      PetscCall(TSSolve(ts, U));
    }
  } else {
    PetscCall(TSSolve(ts, U));
  }
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));

  if (sa == SA_ADJ) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Adjoint model starts here
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*   Set initial conditions for the adjoint integration */
    PetscCall(VecGetArray(lambda[0], &y_ptr));
    y_ptr[0] = 0.0;
    y_ptr[1] = 0.0;
    PetscCall(VecRestoreArray(lambda[0], &y_ptr));

    PetscCall(VecGetArray(mu[0], &x_ptr));
    x_ptr[0] = 0.0;
    PetscCall(VecRestoreArray(mu[0], &x_ptr));

    PetscCall(TSAdjointSolve(ts));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n lambda: d[Psi(tf)]/d[phi0]  d[Psi(tf)]/d[omega0]\n"));
    PetscCall(VecView(lambda[0], PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n mu: d[Psi(tf)]/d[pm]\n"));
    PetscCall(VecView(mu[0], PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(TSGetCostIntegral(ts, &q));
    PetscCall(VecGetArray(q, &x_ptr));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n cost function=%g\n", (double)(x_ptr[0] - ctx.Pm)));
    PetscCall(VecRestoreArray(q, &x_ptr));
    PetscCall(ComputeSensiP(lambda[0], mu[0], &ctx));
    PetscCall(VecGetArray(mu[0], &x_ptr));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n gradient=%g\n", (double)x_ptr[0]));
    PetscCall(VecRestoreArray(mu[0], &x_ptr));
    PetscCall(VecDestroy(&lambda[0]));
    PetscCall(VecDestroy(&mu[0]));
  }
  if (sa == SA_TLM) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n trajectory sensitivity: d[phi(tf)]/d[pm]  d[omega(tf)]/d[pm]\n"));
    PetscCall(MatView(sp, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(TSGetCostIntegral(ts, &q));
    PetscCall(VecGetArray(q, &s_ptr));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n cost function=%g\n", (double)(s_ptr[0] - ctx.Pm)));
    PetscCall(VecRestoreArray(q, &s_ptr));
    PetscCall(MatDenseGetArray(qgrad, &s_ptr));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n gradient=%g\n", (double)s_ptr[0]));
    PetscCall(MatDenseRestoreArray(qgrad, &s_ptr));
    PetscCall(MatDestroy(&qgrad));
    PetscCall(MatDestroy(&sp));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&ctx.Jac));
  PetscCall(MatDestroy(&ctx.Jacp));
  PetscCall(MatDestroy(&ctx.DRDU));
  PetscCall(MatDestroy(&ctx.DRDP));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -sa_method adj -viewer_binary_skip_info -ts_type cn -pc_type lu

   test:
      suffix: 2
      args: -sa_method tlm -ts_type cn -pc_type lu

   test:
      suffix: 3
      args: -sa_method adj -ts_type rk -ts_rk_type 2a -ts_adapt_type dsp

TEST*/
