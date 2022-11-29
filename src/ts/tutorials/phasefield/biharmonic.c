
static char help[] = "Solves biharmonic equation in 1d.\n";

/*
  Solves the equation

    u_t = - kappa  \Delta \Delta u
    Periodic boundary conditions

Evolve the biharmonic heat equation:
---------------
./biharmonic -ts_monitor -snes_monitor   -pc_type lu  -draw_pause .1 -snes_converged_reason  -draw_pause -2   -ts_type cn  -da_refine 5 -mymonitor

Evolve with the restriction that -1 <= u <= 1; i.e. as a variational inequality
---------------
./biharmonic -ts_monitor -snes_monitor   -pc_type lu  -draw_pause .1 -snes_converged_reason  -draw_pause -2   -ts_type cn   -da_refine 5  -mymonitor

   u_t =  kappa \Delta \Delta u +   6.*u*(u_x)^2 + (3*u^2 - 12) \Delta u
    -1 <= u <= 1
    Periodic boundary conditions

Evolve the Cahn-Hillard equations: double well Initial hump shrinks then grows
---------------
./biharmonic -ts_monitor -snes_monitor   -pc_type lu  -draw_pause .1 -snes_converged_reason   -draw_pause -2   -ts_type cn    -da_refine 6   -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard -ts_monitor_draw_solution --mymonitor

Initial hump neither shrinks nor grows when degenerate (otherwise similar solution)

./biharmonic -ts_monitor -snes_monitor   -pc_type lu  -draw_pause .1 -snes_converged_reason   -draw_pause -2   -ts_type cn    -da_refine 6   -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard -degenerate -ts_monitor_draw_solution --mymonitor

./biharmonic -ts_monitor -snes_monitor   -pc_type lu  -draw_pause .1 -snes_converged_reason   -draw_pause -2   -ts_type cn    -da_refine 6   -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard -snes_vi_ignore_function_sign -ts_monitor_draw_solution --mymonitor

Evolve the Cahn-Hillard equations: double obstacle
---------------
./biharmonic -ts_monitor -snes_monitor  -pc_type lu  -draw_pause .1 -snes_converged_reason   -draw_pause -2   -ts_type cn    -da_refine 5   -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard -energy 2 -snes_linesearch_monitor    -ts_monitor_draw_solution --mymonitor

Evolve the Cahn-Hillard equations: logarithmic + double well (never shrinks and then grows)
---------------
./biharmonic -ts_monitor -snes_monitor  -pc_type lu  --snes_converged_reason  -draw_pause -2   -ts_type cn    -da_refine 5   -kappa .0001 -ts_dt 5.96046e-06 -cahn-hillard -energy 3 -snes_linesearch_monitor -theta .00000001    -ts_monitor_draw_solution --ts_max_time 1. -mymonitor

./biharmonic -ts_monitor -snes_monitor  -pc_type lu  --snes_converged_reason  -draw_pause -2   -ts_type cn    -da_refine 5   -kappa .0001 -ts_dt 5.96046e-06 -cahn-hillard -energy 3 -snes_linesearch_monitor -theta .00000001    -ts_monitor_draw_solution --ts_max_time 1. -degenerate -mymonitor

Evolve the Cahn-Hillard equations: logarithmic +  double obstacle (never shrinks, never grows)
---------------
./biharmonic -ts_monitor -snes_monitor  -pc_type lu  --snes_converged_reason  -draw_pause -2   -ts_type cn    -da_refine 5   -kappa .00001 -ts_dt 5.96046e-06 -cahn-hillard -energy 4 -snes_linesearch_monitor -theta .00000001   -ts_monitor_draw_solution --mymonitor

*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>

extern PetscErrorCode FormFunction(TS, PetscReal, Vec, Vec, void *), FormInitialSolution(DM, Vec), MyMonitor(TS, PetscInt, PetscReal, Vec, void *), MyDestroy(void **), FormJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
typedef struct {
  PetscBool           cahnhillard;
  PetscBool           degenerate;
  PetscReal           kappa;
  PetscInt            energy;
  PetscReal           tol;
  PetscReal           theta, theta_c;
  PetscInt            truncation;
  PetscBool           netforce;
  PetscDrawViewPorts *ports;
} UserCtx;

int main(int argc, char **argv)
{
  TS        ts;   /* nonlinear solver */
  Vec       x, r; /* solution, residual vectors */
  Mat       J;    /* Jacobian matrix */
  PetscInt  steps, Mx;
  DM        da;
  PetscReal dt;
  PetscBool mymonitor;
  UserCtx   ctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  ctx.kappa = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-kappa", &ctx.kappa, NULL));
  ctx.degenerate = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-degenerate", &ctx.degenerate, NULL));
  ctx.cahnhillard = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cahn-hillard", &ctx.cahnhillard, NULL));
  ctx.netforce = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-netforce", &ctx.netforce, NULL));
  ctx.energy = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-energy", &ctx.energy, NULL));
  ctx.tol = 1.0e-8;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &ctx.tol, NULL));
  ctx.theta   = .001;
  ctx.theta_c = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta", &ctx.theta, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-theta_c", &ctx.theta_c, NULL));
  ctx.truncation = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-truncation", &ctx.truncation, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-mymonitor", &mymonitor));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 10, 1, 2, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da, 0, "Biharmonic heat equation: u"));
  PetscCall(DMDAGetInfo(da, 0, &Mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  dt = 1.0 / (10. * ctx.kappa * Mx * Mx * Mx * Mx);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &r));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, FormFunction, &ctx));
  PetscCall(DMSetMatType(da, MATAIJ));
  PetscCall(DMCreateMatrix(da, &J));
  PetscCall(TSSetRHSJacobian(ts, J, J, FormJacobian, &ctx));
  PetscCall(TSSetMaxTime(ts, .02));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_INTERPOLATE));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine

     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetType(ts, TSCN));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(da, x));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetSolution(ts, x));

  if (mymonitor) {
    ctx.ports = NULL;
    PetscCall(TSMonitorSet(ts, MyMonitor, &ctx, MyDestroy));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, x));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(VecView(x, PETSC_VIEWER_BINARY_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(TS ts, PetscReal ftime, Vec X, Vec F, void *ptr)
{
  DM           da;
  PetscInt     i, Mx, xs, xm;
  PetscReal    hx, sx;
  PetscScalar *x, *f, c, r, l;
  Vec          localX;
  UserCtx     *ctx = (UserCtx *)ptr;
  PetscReal    tol = ctx->tol, theta = ctx->theta, theta_c = ctx->theta_c, a, b; /* a and b are used in the cubic truncation of the log function */

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 1.0 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da, localX, &x));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (i = xs; i < xs + xm; i++) {
    if (ctx->degenerate) {
      c = (1. - x[i] * x[i]) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
      r = (1. - x[i + 1] * x[i + 1]) * (x[i] + x[i + 2] - 2.0 * x[i + 1]) * sx;
      l = (1. - x[i - 1] * x[i - 1]) * (x[i - 2] + x[i] - 2.0 * x[i - 1]) * sx;
    } else {
      c = (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
      r = (x[i] + x[i + 2] - 2.0 * x[i + 1]) * sx;
      l = (x[i - 2] + x[i] - 2.0 * x[i - 1]) * sx;
    }
    f[i] = -ctx->kappa * (l + r - 2.0 * c) * sx;
    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: /*  double well */
        f[i] += 6. * .25 * x[i] * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (3. * x[i] * x[i] - 1.) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        break;
      case 2: /* double obstacle */
        f[i] += -(x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        break;
      case 3: /* logarithmic + double well */
        f[i] += 6. * .25 * x[i] * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (3. * x[i] * x[i] - 1.) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        if (ctx->truncation == 2) { /* log function with approximated with a quadratic polynomial outside -1.0+2*tol, 1.0-2*tol */
          if (PetscRealPart(x[i]) < -1.0 + 2.0 * tol) f[i] += (.25 * theta / (tol - tol * tol)) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else if (PetscRealPart(x[i]) > 1.0 - 2.0 * tol) f[i] += (.25 * theta / (tol - tol * tol)) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else f[i] += 2.0 * theta * x[i] / ((1.0 - x[i] * x[i]) * (1.0 - x[i] * x[i])) * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (theta / (1.0 - x[i] * x[i])) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        } else { /* log function is approximated with a cubic polynomial outside -1.0+2*tol, 1.0-2*tol */
          a = 2.0 * theta * (1.0 - 2.0 * tol) / (16.0 * tol * tol * (1.0 - tol) * (1.0 - tol));
          b = theta / (4.0 * tol * (1.0 - tol)) - a * (1.0 - 2.0 * tol);
          if (PetscRealPart(x[i]) < -1.0 + 2.0 * tol) f[i] += -1.0 * a * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (-1.0 * a * x[i] + b) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else if (PetscRealPart(x[i]) > 1.0 - 2.0 * tol) f[i] += 1.0 * a * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (a * x[i] + b) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else f[i] += 2.0 * theta * x[i] / ((1.0 - x[i] * x[i]) * (1.0 - x[i] * x[i])) * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (theta / (1.0 - x[i] * x[i])) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        }
        break;
      case 4: /* logarithmic + double obstacle */
        f[i] += -theta_c * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        if (ctx->truncation == 2) { /* quadratic */
          if (PetscRealPart(x[i]) < -1.0 + 2.0 * tol) f[i] += (.25 * theta / (tol - tol * tol)) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else if (PetscRealPart(x[i]) > 1.0 - 2.0 * tol) f[i] += (.25 * theta / (tol - tol * tol)) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else f[i] += 2.0 * theta * x[i] / ((1.0 - x[i] * x[i]) * (1.0 - x[i] * x[i])) * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (theta / (1.0 - x[i] * x[i])) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        } else { /* cubic */
          a = 2.0 * theta * (1.0 - 2.0 * tol) / (16.0 * tol * tol * (1.0 - tol) * (1.0 - tol));
          b = theta / (4.0 * tol * (1.0 - tol)) - a * (1.0 - 2.0 * tol);
          if (PetscRealPart(x[i]) < -1.0 + 2.0 * tol) f[i] += -1.0 * a * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (-1.0 * a * x[i] + b) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else if (PetscRealPart(x[i]) > 1.0 - 2.0 * tol) f[i] += 1.0 * a * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (a * x[i] + b) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
          else f[i] += 2.0 * theta * x[i] / ((1.0 - x[i] * x[i]) * (1.0 - x[i] * x[i])) * .25 * (x[i + 1] - x[i - 1]) * (x[i + 1] - x[i - 1]) * sx + (theta / (1.0 - x[i] * x[i])) * (x[i - 1] + x[i + 1] - 2.0 * x[i]) * sx;
        }
        break;
      }
    }
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormJacobian - Evaluates nonlinear function's Jacobian

*/
PetscErrorCode FormJacobian(TS ts, PetscReal ftime, Vec X, Mat A, Mat B, void *ptr)
{
  DM           da;
  PetscInt     i, Mx, xs, xm;
  MatStencil   row, cols[5];
  PetscReal    hx, sx;
  PetscScalar *x, vals[5];
  Vec          localX;
  UserCtx     *ctx = (UserCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 1.0 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da, localX, &x));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (i = xs; i < xs + xm; i++) {
    row.i = i;
    if (ctx->degenerate) {
      /*PetscScalar c,r,l;
      c = (1. - x[i]*x[i])*(x[i-1] + x[i+1] - 2.0*x[i])*sx;
      r = (1. - x[i+1]*x[i+1])*(x[i] + x[i+2] - 2.0*x[i+1])*sx;
      l = (1. - x[i-1]*x[i-1])*(x[i-2] + x[i] - 2.0*x[i-1])*sx; */
    } else {
      cols[0].i = i - 2;
      vals[0]   = -ctx->kappa * sx * sx;
      cols[1].i = i - 1;
      vals[1]   = 4.0 * ctx->kappa * sx * sx;
      cols[2].i = i;
      vals[2]   = -6.0 * ctx->kappa * sx * sx;
      cols[3].i = i + 1;
      vals[3]   = 4.0 * ctx->kappa * sx * sx;
      cols[4].i = i + 2;
      vals[4]   = -ctx->kappa * sx * sx;
    }
    PetscCall(MatSetValuesStencil(B, 1, &row, 5, cols, vals, INSERT_VALUES));

    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: /* double well */
        /*  f[i] += 6.*.25*x[i]*(x[i+1] - x[i-1])*(x[i+1] - x[i-1])*sx + (3.*x[i]*x[i] - 1.)*(x[i-1] + x[i+1] - 2.0*x[i])*sx; */
        break;
      case 2: /* double obstacle */
        /*        f[i] += -(x[i-1] + x[i+1] - 2.0*x[i])*sx; */
        break;
      case 3: /* logarithmic + double well */
        break;
      case 4: /* logarithmic + double obstacle */
        break;
      }
    }
  }

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayRead(da, localX, &x));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da, Vec U)
{
  PetscInt           i, xs, xm, Mx, N, scale;
  PetscScalar       *u;
  PetscReal          r, hx, x;
  const PetscScalar *f;
  Vec                finesolution;
  PetscViewer        viewer;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 1.0 / (PetscReal)Mx;

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArray(da, U, &u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));

  /*
      Seee heat.c for how to generate InitialSolution.heat
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "InitialSolution.heat", FILE_MODE_READ, &viewer));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &finesolution));
  PetscCall(VecLoad(finesolution, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecGetSize(finesolution, &N));
  scale = N / Mx;
  PetscCall(VecGetArrayRead(finesolution, &f));

  /*
     Compute function over the locally owned part of the grid
  */
  for (i = xs; i < xs + xm; i++) {
    x = i * hx;
    r = PetscSqrtReal((x - .5) * (x - .5));
    if (r < .125) u[i] = 1.0;
    else u[i] = -.5;

    /* With the initial condition above the method is first order in space */
    /* this is a smooth initial condition so the method becomes second order in space */
    /*u[i] = PetscSinScalar(2*PETSC_PI*x); */
    u[i] = f[scale * i];
  }
  PetscCall(VecRestoreArrayRead(finesolution, &f));
  PetscCall(VecDestroy(&finesolution));

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscFunctionReturn(0);
}

/*
    This routine is not parallel
*/
PetscErrorCode MyMonitor(TS ts, PetscInt step, PetscReal time, Vec U, void *ptr)
{
  UserCtx     *ctx = (UserCtx *)ptr;
  PetscDrawLG  lg;
  PetscScalar *u, l, r, c;
  PetscInt     Mx, i, xs, xm, cnt;
  PetscReal    x, y, hx, pause, sx, len, max, xx[4], yy[4], xx_netforce, yy_netforce, yup, ydown, y2, len2;
  PetscDraw    draw;
  Vec          localU;
  DM           da;
  int          colors[] = {PETSC_DRAW_YELLOW, PETSC_DRAW_RED, PETSC_DRAW_BLUE, PETSC_DRAW_PLUM, PETSC_DRAW_BLACK};
  /*
  const char *const  legend[3][3] = {{"-kappa (\\grad u,\\grad u)","(1 - u^2)^2"},{"-kappa (\\grad u,\\grad u)","(1 - u^2)"},{"-kappa (\\grad u,\\grad u)","logarithmic"}};
   */
  PetscDrawAxis       axis;
  PetscDrawViewPorts *ports;
  PetscReal           tol = ctx->tol, theta = ctx->theta, theta_c = ctx->theta_c, a, b; /* a and b are used in the cubic truncation of the log function */
  PetscReal           vbounds[] = {-1.1, 1.1};

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawSetBounds(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD), 1, vbounds));
  PetscCall(PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD), 800, 600));
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
  hx = 1.0 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU));
  PetscCall(DMDAVecGetArrayRead(da, localU, &u));

  PetscCall(PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD), 1, &lg));
  PetscCall(PetscDrawLGGetDraw(lg, &draw));
  PetscCall(PetscDrawCheckResizedWindow(draw));
  if (!ctx->ports) PetscCall(PetscDrawViewPortsCreateRect(draw, 1, 3, &ctx->ports));
  ports = ctx->ports;
  PetscCall(PetscDrawLGGetAxis(lg, &axis));
  PetscCall(PetscDrawLGReset(lg));

  xx[0] = 0.0;
  xx[1] = 1.0;
  cnt   = 2;
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-zoom", xx, &cnt, NULL));
  xs = xx[0] / hx;
  xm = (xx[1] - xx[0]) / hx;

  /*
      Plot the  energies
  */
  PetscCall(PetscDrawLGSetDimension(lg, 1 + (ctx->cahnhillard ? 1 : 0) + (ctx->energy == 3)));
  PetscCall(PetscDrawLGSetColors(lg, colors + 1));
  PetscCall(PetscDrawViewPortsSet(ports, 2));
  x = hx * xs;
  for (i = xs; i < xs + xm; i++) {
    xx[0] = xx[1] = xx[2] = x;
    if (ctx->degenerate) yy[0] = PetscRealPart(.25 * (1. - u[i] * u[i]) * ctx->kappa * (u[i - 1] - u[i + 1]) * (u[i - 1] - u[i + 1]) * sx);
    else yy[0] = PetscRealPart(.25 * ctx->kappa * (u[i - 1] - u[i + 1]) * (u[i - 1] - u[i + 1]) * sx);

    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: /* double well */
        yy[1] = .25 * PetscRealPart((1. - u[i] * u[i]) * (1. - u[i] * u[i]));
        break;
      case 2: /* double obstacle */
        yy[1] = .5 * PetscRealPart(1. - u[i] * u[i]);
        break;
      case 3: /* logarithm + double well */
        yy[1] = .25 * PetscRealPart((1. - u[i] * u[i]) * (1. - u[i] * u[i]));
        if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) yy[2] = .5 * theta * (2.0 * tol * PetscLogReal(tol) + PetscRealPart(1.0 - u[i]) * PetscLogReal(PetscRealPart(1. - u[i]) / 2.0));
        else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) yy[2] = .5 * theta * (PetscRealPart(1.0 + u[i]) * PetscLogReal(PetscRealPart(1.0 + u[i]) / 2.0) + 2.0 * tol * PetscLogReal(tol));
        else yy[2] = .5 * theta * (PetscRealPart(1.0 + u[i]) * PetscLogReal(PetscRealPart(1.0 + u[i]) / 2.0) + PetscRealPart(1.0 - u[i]) * PetscLogReal(PetscRealPart(1.0 - u[i]) / 2.0));
        break;
      case 4: /* logarithm + double obstacle */
        yy[1] = .5 * theta_c * PetscRealPart(1.0 - u[i] * u[i]);
        if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) yy[2] = .5 * theta * (2.0 * tol * PetscLogReal(tol) + PetscRealPart(1.0 - u[i]) * PetscLogReal(PetscRealPart(1. - u[i]) / 2.0));
        else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) yy[2] = .5 * theta * (PetscRealPart(1.0 + u[i]) * PetscLogReal(PetscRealPart(1.0 + u[i]) / 2.0) + 2.0 * tol * PetscLogReal(tol));
        else yy[2] = .5 * theta * (PetscRealPart(1.0 + u[i]) * PetscLogReal(PetscRealPart(1.0 + u[i]) / 2.0) + PetscRealPart(1.0 - u[i]) * PetscLogReal(PetscRealPart(1.0 - u[i]) / 2.0));
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "It will always be one of the values");
      }
    }
    PetscCall(PetscDrawLGAddPoint(lg, xx, yy));
    x += hx;
  }
  PetscCall(PetscDrawGetPause(draw, &pause));
  PetscCall(PetscDrawSetPause(draw, 0.0));
  PetscCall(PetscDrawAxisSetLabels(axis, "Energy", "", ""));
  /*  PetscCall(PetscDrawLGSetLegend(lg,legend[ctx->energy-1])); */
  PetscCall(PetscDrawLGDraw(lg));

  /*
      Plot the  forces
  */
  PetscCall(PetscDrawLGSetDimension(lg, 0 + (ctx->cahnhillard ? 2 : 0) + (ctx->energy == 3)));
  PetscCall(PetscDrawLGSetColors(lg, colors + 1));
  PetscCall(PetscDrawViewPortsSet(ports, 1));
  PetscCall(PetscDrawLGReset(lg));
  x   = xs * hx;
  max = 0.;
  for (i = xs; i < xs + xm; i++) {
    xx[0] = xx[1] = xx[2] = xx[3] = x;
    xx_netforce                   = x;
    if (ctx->degenerate) {
      c = (1. - u[i] * u[i]) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
      r = (1. - u[i + 1] * u[i + 1]) * (u[i] + u[i + 2] - 2.0 * u[i + 1]) * sx;
      l = (1. - u[i - 1] * u[i - 1]) * (u[i - 2] + u[i] - 2.0 * u[i - 1]) * sx;
    } else {
      c = (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
      r = (u[i] + u[i + 2] - 2.0 * u[i + 1]) * sx;
      l = (u[i - 2] + u[i] - 2.0 * u[i - 1]) * sx;
    }
    yy[0]       = PetscRealPart(-ctx->kappa * (l + r - 2.0 * c) * sx);
    yy_netforce = yy[0];
    max         = PetscMax(max, PetscAbs(yy[0]));
    if (ctx->cahnhillard) {
      switch (ctx->energy) {
      case 1: /* double well */
        yy[1] = PetscRealPart(6. * .25 * u[i] * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (3. * u[i] * u[i] - 1.) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
        break;
      case 2: /* double obstacle */
        yy[1] = -PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
        break;
      case 3: /* logarithmic + double well */
        yy[1] = PetscRealPart(6. * .25 * u[i] * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (3. * u[i] * u[i] - 1.) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
        if (ctx->truncation == 2) { /* quadratic */
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) yy[2] = (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) yy[2] = (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
          else yy[2] = PetscRealPart(2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
        } else { /* cubic */
          a = 2.0 * theta * (1.0 - 2.0 * tol) / (16.0 * tol * tol * (1.0 - tol) * (1.0 - tol));
          b = theta / (4.0 * tol * (1.0 - tol)) - a * (1.0 - 2.0 * tol);
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) yy[2] = PetscRealPart(-1.0 * a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (-1.0 * a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) yy[2] = PetscRealPart(1.0 * a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
          else yy[2] = PetscRealPart(2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
        }
        break;
      case 4: /* logarithmic + double obstacle */
        yy[1] = theta_c * PetscRealPart(-(u[i - 1] + u[i + 1] - 2.0 * u[i])) * sx;
        if (ctx->truncation == 2) {
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) yy[2] = (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) yy[2] = (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
          else yy[2] = PetscRealPart(2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
        } else {
          a = 2.0 * theta * (1.0 - 2.0 * tol) / (16.0 * tol * tol * (1.0 - tol) * (1.0 - tol));
          b = theta / (4.0 * tol * (1.0 - tol)) - a * (1.0 - 2.0 * tol);
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) yy[2] = PetscRealPart(-1.0 * a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (-1.0 * a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) yy[2] = PetscRealPart(1.0 * a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
          else yy[2] = PetscRealPart(2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx);
        }
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "It will always be one of the values");
      }
      if (ctx->energy < 3) {
        max         = PetscMax(max, PetscAbs(yy[1]));
        yy[2]       = yy[0] + yy[1];
        yy_netforce = yy[2];
      } else {
        max         = PetscMax(max, PetscAbs(yy[1] + yy[2]));
        yy[3]       = yy[0] + yy[1] + yy[2];
        yy_netforce = yy[3];
      }
    }
    if (ctx->netforce) {
      PetscCall(PetscDrawLGAddPoint(lg, &xx_netforce, &yy_netforce));
    } else {
      PetscCall(PetscDrawLGAddPoint(lg, xx, yy));
    }
    x += hx;
    /*if (max > 7200150000.0) */
    /* printf("max very big when i = %d\n",i); */
  }
  PetscCall(PetscDrawAxisSetLabels(axis, "Right hand side", "", ""));
  PetscCall(PetscDrawLGSetLegend(lg, NULL));
  PetscCall(PetscDrawLGDraw(lg));

  /*
        Plot the solution
  */
  PetscCall(PetscDrawLGSetDimension(lg, 1));
  PetscCall(PetscDrawViewPortsSet(ports, 0));
  PetscCall(PetscDrawLGReset(lg));
  x = hx * xs;
  PetscCall(PetscDrawLGSetLimits(lg, x, x + (xm - 1) * hx, -1.1, 1.1));
  PetscCall(PetscDrawLGSetColors(lg, colors));
  for (i = xs; i < xs + xm; i++) {
    xx[0] = x;
    yy[0] = PetscRealPart(u[i]);
    PetscCall(PetscDrawLGAddPoint(lg, xx, yy));
    x += hx;
  }
  PetscCall(PetscDrawAxisSetLabels(axis, "Solution", "", ""));
  PetscCall(PetscDrawLGDraw(lg));

  /*
      Print the  forces as arrows on the solution
  */
  x   = hx * xs;
  cnt = xm / 60;
  cnt = (!cnt) ? 1 : cnt;

  for (i = xs; i < xs + xm; i += cnt) {
    y = yup = ydown = PetscRealPart(u[i]);
    c               = (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx;
    r               = (u[i] + u[i + 2] - 2.0 * u[i + 1]) * sx;
    l               = (u[i - 2] + u[i] - 2.0 * u[i - 1]) * sx;
    len             = -.5 * PetscRealPart(ctx->kappa * (l + r - 2.0 * c) * sx) / max;
    PetscCall(PetscDrawArrow(draw, x, y, x, y + len, PETSC_DRAW_RED));
    if (ctx->cahnhillard) {
      if (len < 0.) ydown += len;
      else yup += len;

      switch (ctx->energy) {
      case 1: /* double well */
        len = .5 * PetscRealPart(6. * .25 * u[i] * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (3. * u[i] * u[i] - 1.) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max;
        break;
      case 2: /* double obstacle */
        len = -.5 * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx / max;
        break;
      case 3: /* logarithmic + double well */
        len = .5 * PetscRealPart(6. * .25 * u[i] * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (3. * u[i] * u[i] - 1.) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max;
        if (len < 0.) ydown += len;
        else yup += len;

        if (ctx->truncation == 2) { /* quadratic */
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) len2 = .5 * (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx / max;
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) len2 = .5 * (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx / max;
          else len2 = PetscRealPart(.5 * (2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max);
        } else { /* cubic */
          a = 2.0 * theta * (1.0 - 2.0 * tol) / (16.0 * tol * tol * (1.0 - tol) * (1.0 - tol));
          b = theta / (4.0 * tol * (1.0 - tol)) - a * (1.0 - 2.0 * tol);
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) len2 = PetscRealPart(.5 * (-1.0 * a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (-1.0 * a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max);
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) len2 = PetscRealPart(.5 * (a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max);
          else len2 = PetscRealPart(.5 * (2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max);
        }
        y2 = len < 0 ? ydown : yup;
        PetscCall(PetscDrawArrow(draw, x, y2, x, y2 + len2, PETSC_DRAW_PLUM));
        break;
      case 4: /* logarithmic + double obstacle */
        len = -.5 * theta_c * PetscRealPart(-(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx / max);
        if (len < 0.) ydown += len;
        else yup += len;

        if (ctx->truncation == 2) { /* quadratic */
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) len2 = .5 * (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx / max;
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) len2 = .5 * (.25 * theta / (tol - tol * tol)) * PetscRealPart(u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx / max;
          else len2 = PetscRealPart(.5 * (2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max);
        } else { /* cubic */
          a = 2.0 * theta * (1.0 - 2.0 * tol) / (16.0 * tol * tol * (1.0 - tol) * (1.0 - tol));
          b = theta / (4.0 * tol * (1.0 - tol)) - a * (1.0 - 2.0 * tol);
          if (PetscRealPart(u[i]) < -1.0 + 2.0 * tol) len2 = .5 * PetscRealPart(-1.0 * a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (-1.0 * a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max;
          else if (PetscRealPart(u[i]) > 1.0 - 2.0 * tol) len2 = .5 * PetscRealPart(a * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (a * u[i] + b) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max;
          else len2 = .5 * PetscRealPart(2.0 * theta * u[i] / ((1.0 - u[i] * u[i]) * (1.0 - u[i] * u[i])) * .25 * (u[i + 1] - u[i - 1]) * (u[i + 1] - u[i - 1]) * sx + (theta / (1.0 - u[i] * u[i])) * (u[i - 1] + u[i + 1] - 2.0 * u[i]) * sx) / max;
        }
        y2 = len < 0 ? ydown : yup;
        PetscCall(PetscDrawArrow(draw, x, y2, x, y2 + len2, PETSC_DRAW_PLUM));
        break;
      }
      PetscCall(PetscDrawArrow(draw, x, y, x, y + len, PETSC_DRAW_BLUE));
    }
    x += cnt * hx;
  }
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &x));
  PetscCall(DMRestoreLocalVector(da, &localU));
  PetscCall(PetscDrawStringSetSize(draw, .2, .2));
  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawSetPause(draw, pause));
  PetscCall(PetscDrawPause(draw));
  PetscFunctionReturn(0);
}

PetscErrorCode MyDestroy(void **ptr)
{
  UserCtx *ctx = *(UserCtx **)ptr;

  PetscFunctionBegin;
  PetscCall(PetscDrawViewPortsDestroy(ctx->ports));
  PetscFunctionReturn(0);
}

/*TEST

   test:
     TODO: currently requires initial condition file generated by heat

TEST*/
