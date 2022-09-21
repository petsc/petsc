static char help[] = "Demonstrates adjoint sensitivity analysis for Reaction-Diffusion Equations.\n";

/*
  See ex5.c for details on the equation.
  This code demonestrates the TSAdjoint interface to a system of time-dependent partial differential equations.
  It computes the sensitivity of a component in the final solution, which locates in the center of the 2D domain, w.r.t. the initial conditions.
  The user does not need to provide any additional functions. The required functions in the original simulation are reused in the adjoint run.

  Runtime options:
    -forwardonly  - run the forward simulation without adjoint
    -implicitform - provide IFunction and IJacobian to TS, if not set, RHSFunction and RHSJacobian will be used
    -aijpc        - set the preconditioner matrix to be aij (the Jacobian matrix can be of a different type such as ELL)
*/

#include "reaction_diffusion.h"
#include <petscdm.h>
#include <petscdmda.h>

/* Matrix free support */
typedef struct {
  PetscReal time;
  Vec       U;
  Vec       Udot;
  PetscReal shift;
  AppCtx   *appctx;
  TS        ts;
} MCtx;

/*
   User-defined routines
*/
PetscErrorCode InitialConditions(DM, Vec);
PetscErrorCode RHSJacobianShell(TS, PetscReal, Vec, Mat, Mat, void *);
PetscErrorCode IJacobianShell(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);

PetscErrorCode InitializeLambda(DM da, Vec lambda, PetscReal x, PetscReal y)
{
  PetscInt i, j, Mx, My, xs, ys, xm, ym;
  Field  **l;
  PetscFunctionBegin;

  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  /* locate the global i index for x and j index for y */
  i = (PetscInt)(x * (Mx - 1));
  j = (PetscInt)(y * (My - 1));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  if (xs <= i && i < xs + xm && ys <= j && j < ys + ym) {
    /* the i,j vertex is on this process */
    PetscCall(DMDAVecGetArray(da, lambda, &l));
    l[j][i].u = 1.0;
    l[j][i].v = 1.0;
    PetscCall(DMDAVecRestoreArray(da, lambda, &l));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MyRHSMatMultTranspose(Mat A_shell, Vec X, Vec Y)
{
  MCtx       *mctx;
  AppCtx     *appctx;
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   hx, hy, sx, sy;
  PetscScalar uc, uxx, uyy, vc, vxx, vyy, ucb, vcb;
  Field     **u, **x, **y;
  Vec         localX;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A_shell, &mctx));
  appctx = mctx->appctx;
  PetscCall(TSGetDM(mctx->ts, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 2.50 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  hy = 2.50 / (PetscReal)My;
  sy = 1.0 / (hy * hy);

  /* Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition. */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da, localX, &x));
  PetscCall(DMDAVecGetArrayRead(da, mctx->U, &u));
  PetscCall(DMDAVecGetArray(da, Y, &y));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /* Compute function over the locally owned part of the grid */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      uc        = u[j][i].u;
      ucb       = x[j][i].u;
      uxx       = (-2.0 * ucb + x[j][i - 1].u + x[j][i + 1].u) * sx;
      uyy       = (-2.0 * ucb + x[j - 1][i].u + x[j + 1][i].u) * sy;
      vc        = u[j][i].v;
      vcb       = x[j][i].v;
      vxx       = (-2.0 * vcb + x[j][i - 1].v + x[j][i + 1].v) * sx;
      vyy       = (-2.0 * vcb + x[j - 1][i].v + x[j + 1][i].v) * sy;
      y[j][i].u = appctx->D1 * (uxx + uyy) - ucb * (vc * vc + appctx->gamma) + vc * vc * vcb;
      y[j][i].v = appctx->D2 * (vxx + vyy) - 2.0 * uc * vc * ucb + (2.0 * uc * vc - appctx->gamma - appctx->kappa) * vcb;
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMDAVecRestoreArrayRead(da, mctx->U, &u));
  PetscCall(DMDAVecRestoreArray(da, Y, &y));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscFunctionReturn(0);
}

static PetscErrorCode MyIMatMultTranspose(Mat A_shell, Vec X, Vec Y)
{
  MCtx       *mctx;
  AppCtx     *appctx;
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   hx, hy, sx, sy;
  PetscScalar uc, uxx, uyy, vc, vxx, vyy, ucb, vcb;
  Field     **u, **x, **y;
  Vec         localX;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A_shell, &mctx));
  appctx = mctx->appctx;
  PetscCall(TSGetDM(mctx->ts, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 2.50 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  hy = 2.50 / (PetscReal)My;
  sy = 1.0 / (hy * hy);

  /* Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition. */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da, localX, &x));
  PetscCall(DMDAVecGetArrayRead(da, mctx->U, &u));
  PetscCall(DMDAVecGetArray(da, Y, &y));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /* Compute function over the locally owned part of the grid */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      uc        = u[j][i].u;
      ucb       = x[j][i].u;
      uxx       = (-2.0 * ucb + x[j][i - 1].u + x[j][i + 1].u) * sx;
      uyy       = (-2.0 * ucb + x[j - 1][i].u + x[j + 1][i].u) * sy;
      vc        = u[j][i].v;
      vcb       = x[j][i].v;
      vxx       = (-2.0 * vcb + x[j][i - 1].v + x[j][i + 1].v) * sx;
      vyy       = (-2.0 * vcb + x[j - 1][i].v + x[j + 1][i].v) * sy;
      y[j][i].u = appctx->D1 * (uxx + uyy) - ucb * (vc * vc + appctx->gamma) + vc * vc * vcb;
      y[j][i].v = appctx->D2 * (vxx + vyy) - 2.0 * uc * vc * ucb + (2.0 * uc * vc - appctx->gamma - appctx->kappa) * vcb;
      y[j][i].u = mctx->shift * ucb - y[j][i].u;
      y[j][i].v = mctx->shift * vcb - y[j][i].v;
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMDAVecRestoreArrayRead(da, mctx->U, &u));
  PetscCall(DMDAVecRestoreArray(da, Y, &y));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscFunctionReturn(0);
}

static PetscErrorCode MyIMatMult(Mat A_shell, Vec X, Vec Y)
{
  MCtx       *mctx;
  AppCtx     *appctx;
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   hx, hy, sx, sy;
  PetscScalar uc, uxx, uyy, vc, vxx, vyy, ucb, vcb;
  Field     **u, **x, **y;
  Vec         localX;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A_shell, &mctx));
  appctx = mctx->appctx;
  PetscCall(TSGetDM(mctx->ts, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 2.50 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  hy = 2.50 / (PetscReal)My;
  sy = 1.0 / (hy * hy);

  /* Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition. */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da, localX, &x));
  PetscCall(DMDAVecGetArrayRead(da, mctx->U, &u));
  PetscCall(DMDAVecGetArray(da, Y, &y));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /* Compute function over the locally owned part of the grid */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      uc        = u[j][i].u;
      ucb       = x[j][i].u;
      uxx       = (-2.0 * ucb + x[j][i - 1].u + x[j][i + 1].u) * sx;
      uyy       = (-2.0 * ucb + x[j - 1][i].u + x[j + 1][i].u) * sy;
      vc        = u[j][i].v;
      vcb       = x[j][i].v;
      vxx       = (-2.0 * vcb + x[j][i - 1].v + x[j][i + 1].v) * sx;
      vyy       = (-2.0 * vcb + x[j - 1][i].v + x[j + 1][i].v) * sy;
      y[j][i].u = appctx->D1 * (uxx + uyy) - (vc * vc + appctx->gamma) * ucb - 2.0 * uc * vc * vcb;
      y[j][i].v = appctx->D2 * (vxx + vyy) + vc * vc * ucb + (2.0 * uc * vc - appctx->gamma - appctx->kappa) * vcb;
      y[j][i].u = mctx->shift * ucb - y[j][i].u;
      y[j][i].v = mctx->shift * vcb - y[j][i].v;
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMDAVecRestoreArrayRead(da, mctx->U, &u));
  PetscCall(DMDAVecRestoreArray(da, Y, &y));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS             ts; /* ODE integrator */
  Vec            x;  /* solution */
  DM             da;
  AppCtx         appctx;
  MCtx           mctx;
  Vec            lambda[1];
  PetscBool      forwardonly = PETSC_FALSE, implicitform = PETSC_TRUE, mf = PETSC_FALSE;
  PetscLogDouble v1, v2;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-forwardonly", &forwardonly, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-implicitform", &implicitform, NULL));
  appctx.aijpc = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-aijpc", &appctx.aijpc, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mf", &mf, NULL));

  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;

  PetscLogStageRegister("MyAdjoint", &stage);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, 64, 64, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da, 0, "u"));
  PetscCall(DMDASetFieldName(da, 1, "v"));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetEquationType(ts, TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (!implicitform) {
    PetscCall(TSSetType(ts, TSRK));
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &appctx));
    PetscCall(TSSetRHSJacobian(ts, NULL, NULL, RHSJacobian, &appctx));
  } else {
    PetscCall(TSSetType(ts, TSCN));
    PetscCall(TSSetIFunction(ts, NULL, IFunction, &appctx));
    if (appctx.aijpc) {
      Mat A, B;

      PetscCall(DMSetMatType(da, MATSELL));
      PetscCall(DMCreateMatrix(da, &A));
      PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &B));
      /* FIXME do we need to change viewer to display matrix in natural ordering as DMCreateMatrix_DA does? */
      PetscCall(TSSetIJacobian(ts, A, B, IJacobian, &appctx));
      PetscCall(MatDestroy(&A));
      PetscCall(MatDestroy(&B));
    } else {
      PetscCall(TSSetIJacobian(ts, NULL, NULL, IJacobian, &appctx));
    }
  }

  if (mf) {
    PetscInt xm, ym, Mx, My, dof;
    mctx.ts     = ts;
    mctx.appctx = &appctx;
    PetscCall(VecDuplicate(x, &mctx.U));
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Create matrix free context
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, &dof, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
    PetscCall(DMDAGetCorners(da, NULL, NULL, NULL, &xm, &ym, NULL));
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, dof * xm * ym, PETSC_DETERMINE, dof * Mx * My, dof * Mx * My, &mctx, &appctx.A));
    PetscCall(MatShellSetOperation(appctx.A, MATOP_MULT_TRANSPOSE, (void (*)(void))MyRHSMatMultTranspose));
    if (!implicitform) { /* for explicit methods only */
      PetscCall(TSSetRHSJacobian(ts, appctx.A, appctx.A, RHSJacobianShell, &appctx));
    } else {
      /* PetscCall(VecDuplicate(appctx.U,&mctx.Udot)); */
      PetscCall(MatShellSetOperation(appctx.A, MATOP_MULT, (void (*)(void))MyIMatMult));
      PetscCall(MatShellSetOperation(appctx.A, MATOP_MULT_TRANSPOSE, (void (*)(void))MyIMatMultTranspose));
      PetscCall(TSSetIJacobian(ts, appctx.A, appctx.A, IJacobianShell, &appctx));
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(InitialConditions(da, x));
  PetscCall(TSSetSolution(ts, x));

  /*
    Have the TS save its trajectory so that TSAdjointSolve() may be used
  */
  if (!forwardonly) PetscCall(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts, 200.0));
  PetscCall(TSSetTimeStep(ts, 0.5));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(PetscTime(&v1));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, x));
  if (!forwardonly) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Start the Adjoint model
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(VecDuplicate(x, &lambda[0]));
    /*   Reset initial conditions for the adjoint integration */
    PetscCall(InitializeLambda(da, lambda[0], 0.5, 0.5));
    PetscCall(TSSetCostGradients(ts, 1, lambda, NULL));
    PetscLogStagePush(stage);
    PetscCall(TSAdjointSolve(ts));
    PetscLogStagePop();
    PetscCall(VecDestroy(&lambda[0]));
  }
  PetscCall(PetscTime(&v2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %.3lf ", v2 - v1));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  if (mf) {
    PetscCall(VecDestroy(&mctx.U));
    /* PetscCall(VecDestroy(&mctx.Udot));*/
    PetscCall(MatDestroy(&appctx.A));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */
PetscErrorCode RHSJacobianShell(TS ts, PetscReal t, Vec U, Mat A, Mat BB, void *ctx)
{
  MCtx *mctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &mctx));
  PetscCall(VecCopy(U, mctx->U));
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobianShell(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat A, Mat BB, void *ctx)
{
  MCtx *mctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &mctx));
  PetscCall(VecCopy(U, mctx->U));
  /* PetscCall(VecCopy(Udot,mctx->Udot)); */
  mctx->shift = a;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode InitialConditions(DM da, Vec U)
{
  PetscInt  i, j, xs, ys, xm, ym, Mx, My;
  Field   **u;
  PetscReal hx, hy, x, y;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 2.5 / (PetscReal)Mx;
  hy = 2.5 / (PetscReal)My;

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArray(da, U, &u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    y = j * hy;
    for (i = xs; i < xs + xm; i++) {
      x = i * hx;
      if ((1.0 <= x) && (x <= 1.5) && (1.0 <= y) && (y <= 1.5)) u[j][i].v = .25 * PetscPowReal(PetscSinReal(4.0 * PETSC_PI * x), 2.0) * PetscPowReal(PetscSinReal(4.0 * PETSC_PI * y), 2.0);
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0 * u[j][i].v;
    }
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      depends: reaction_diffusion.c
      requires: !complex !single

   test:
      requires: cams
      args: -mf -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -ts_trajectory_type memory -ts_trajectory_solution_only 0 -ts_trajectory_max_units_ram 6 -ts_trajectory_memory_type cams
TEST*/
