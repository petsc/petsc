
static char help[] = "Finds optimal parameter P_m for the generator system while maintaining generator stability.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

F*/

/*
  Solve the same optimization problem as in ex3opt.c.
  Use finite difference to approximate the gradients.
*/
#include <petsctao.h>
#include <petscts.h>
#include "ex3.h"

PetscErrorCode FormFunction(Tao, Vec, PetscReal *, void *);

PetscErrorCode monitor(Tao tao, AppCtx *ctx)
{
  FILE              *fp;
  PetscInt           iterate;
  PetscReal          f, gnorm, cnorm, xdiff;
  Vec                X, G;
  const PetscScalar *x, *g;
  TaoConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(TaoGetSolutionStatus(tao, &iterate, &f, &gnorm, &cnorm, &xdiff, &reason));
  PetscCall(TaoGetSolution(tao, &X));
  PetscCall(TaoGetGradient(tao, &G, NULL, NULL));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(G, &g));
  fp = fopen("ex3opt_fd_conv.out", "a");
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "%" PetscInt_FMT " %g %.12lf %.12lf\n", iterate, (double)gnorm, (double)PetscRealPart(x[0]), (double)PetscRealPart(g[0])));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(G, &g));
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec          p;
  PetscScalar *x_ptr;
  PetscMPIInt  size;
  AppCtx       ctx;
  Vec          lowerb, upperb;
  Tao          tao;
  KSP          ksp;
  PC           pc;
  PetscBool    printtofile;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

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
    ctx.Pm = 1.06;
    PetscCall(PetscOptionsScalar("-Pm", "", "", ctx.Pm, &ctx.Pm, NULL));
    ctx.tf  = 0.1;
    ctx.tcl = 0.2;
    PetscCall(PetscOptionsReal("-tf", "Time to start fault", "", ctx.tf, &ctx.tf, NULL));
    PetscCall(PetscOptionsReal("-tcl", "Time to end fault", "", ctx.tcl, &ctx.tcl, NULL));
    printtofile = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-printtofile", "Print convergence results to file", "", printtofile, &printtofile, NULL));
  }
  PetscOptionsEnd();

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBLMVM));
  if (printtofile) PetscCall(TaoSetMonitor(tao, (PetscErrorCode(*)(Tao, void *))monitor, (void *)&ctx, NULL));
  PetscCall(TaoSetMaximumIterations(tao, 30));
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 1, &p));
  PetscCall(VecGetArray(p, &x_ptr));
  x_ptr[0] = ctx.Pm;
  PetscCall(VecRestoreArray(p, &x_ptr));

  PetscCall(TaoSetSolution(tao, p));
  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjective(tao, FormFunction, (void *)&ctx));
  PetscCall(TaoSetGradient(tao, NULL, TaoDefaultComputeGradient, (void *)&ctx));

  /* Set bounds for the optimization */
  PetscCall(VecDuplicate(p, &lowerb));
  PetscCall(VecDuplicate(p, &upperb));
  PetscCall(VecGetArray(lowerb, &x_ptr));
  x_ptr[0] = 0.;
  PetscCall(VecRestoreArray(lowerb, &x_ptr));
  PetscCall(VecGetArray(upperb, &x_ptr));
  x_ptr[0] = 1.1;
  PetscCall(VecRestoreArray(upperb, &x_ptr));
  PetscCall(TaoSetVariableBounds(tao, lowerb, upperb));

  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoGetKSP(tao, &ksp));
  if (ksp) {
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCNONE));
  }

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  PetscCall(VecView(p, PETSC_VIEWER_STDOUT_WORLD));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&lowerb));
  PetscCall(VecDestroy(&upperb));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------ */
/*
   FormFunction - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
*/
PetscErrorCode FormFunction(Tao tao, Vec P, PetscReal *f, void *ctx0)
{
  AppCtx            *ctx = (AppCtx *)ctx0;
  TS                 ts, quadts;
  Vec                U; /* solution will be stored here */
  Mat                A; /* Jacobian matrix */
  PetscInt           n = 2;
  PetscReal          ftime;
  PetscInt           steps;
  PetscScalar       *u;
  const PetscScalar *x_ptr, *qx_ptr;
  Vec                q;
  PetscInt           direction[2];
  PetscBool          terminate[2];

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(P, &x_ptr));
  ctx->Pm = x_ptr[0];
  PetscCall(VecRestoreArrayRead(P, &x_ptr));
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
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSCN));
  PetscCall(TSSetIFunction(ts, NULL, (TSIFunction)IFunction, ctx));
  PetscCall(TSSetIJacobian(ts, A, A, (TSIJacobian)IJacobian, ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(U, &u));
  u[0] = PetscAsinScalar(ctx->Pm / ctx->Pmax);
  u[1] = 1.0;
  PetscCall(VecRestoreArray(U, &u));
  PetscCall(TSSetSolution(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, 0.03125));
  PetscCall(TSCreateQuadratureTS(ts, PETSC_TRUE, &quadts));
  PetscCall(TSGetSolution(quadts, &q));
  PetscCall(VecSet(q, 0.0));
  PetscCall(TSSetRHSFunction(quadts, NULL, (TSRHSFunction)CostIntegrand, ctx));
  PetscCall(TSSetFromOptions(ts));

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;

  PetscCall(TSSetEventHandler(ts, 2, direction, terminate, EventFunction, PostEventFunction, (void *)ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, U));

  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(VecGetArrayRead(q, &qx_ptr));
  *f = -ctx->Pm + qx_ptr[0];
  PetscCall(VecRestoreArrayRead(q, &qx_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -ts_type cn -pc_type lu -tao_monitor -tao_gatol 1e-3

TEST*/
