#pragma once

#include <petscdmda.h>
#include <petscts.h>

/* Helpers are `static inline` for small math; bare `static` for address-taken callbacks
   (e.g. ShallowWaterRHS2D registered via TSSetRHSFunction) and for one-shot setup helpers
   that are intentionally duplicated per translation unit (tutorial isolation - ex4.c and
   ex4fwd.c stay independent). */

/* Wet/dry threshold: cells with h below this fall back to zero flux to avoid division by ~0. */
#define EX4_DRY_TOL 1e-10

typedef enum {
  EX4_FLUX_RUSANOV
} Ex4FluxType;

static const char *const Ex4FluxTypes[] = {"rusanov", "Ex4FluxType", "EX4_FLUX_", NULL};

typedef struct {
  DM          da;
  PetscInt    nx, ny;
  PetscReal   Lx, Ly;
  PetscReal   dx, dy;
  PetscReal   g;
  PetscReal   dt;
  PetscReal   t_current; /* cumulative physical time tracked by ShallowWaterStep2D() (matrix variant); the MMS source uses the per-call t_start argument of ShallowWaterStep2DVec() */
  TS          ts;
  PetscReal   h0;
  PetscReal   Ax, Ay; /* wave amplitudes in x and y */
  PetscBool   verify_mms;
  Ex4FluxType flux_type;
} ShallowWater2DCtx;

static inline void ManufacturedSolution2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal h0, PetscReal A, PetscReal *h, PetscReal *hu, PetscReal *hv)
{
  PetscReal sx = PetscSinReal(2.0 * PETSC_PI * x / Lx);
  PetscReal sy = PetscSinReal(2.0 * PETSC_PI * y / Ly);
  PetscReal cx = PetscCosReal(2.0 * PETSC_PI * x / Lx);
  PetscReal cy = PetscCosReal(2.0 * PETSC_PI * y / Ly);

  *h  = h0 + A * PetscSinReal(t) * sx * sy;
  *hu = A * PetscCosReal(t) * cx * sy;
  *hv = A * PetscSinReal(t) * sx * cy;
}

/* MMS uses a single scalar amplitude A: callers pass sw->Ax with the invariant Ax == Ay enforced
   by SetInitialCondition() before the integration begins. ShallowWater2DContextCreate()
   collectively enforces h0 - A > EX4_DRY_TOL when verify_mms is enabled (h = h0 + A*sin(t)*sx*sy
   ranges in [h0-A, h0+A] and the source divides by h*h), so this hot path skips the per-cell guard. */
static inline void ManufacturedSource2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal g, PetscReal h0, PetscReal A, PetscReal *S_h, PetscReal *S_hu, PetscReal *S_hv)
{
  PetscReal h, hu, hv;
  PetscReal kx  = 2.0 * PETSC_PI / Lx;
  PetscReal ky  = 2.0 * PETSC_PI / Ly;
  PetscReal sx  = PetscSinReal(kx * x);
  PetscReal sy  = PetscSinReal(ky * y);
  PetscReal cx  = PetscCosReal(kx * x);
  PetscReal cy  = PetscCosReal(ky * y);
  PetscReal ht  = A * PetscCosReal(t) * sx * sy;
  PetscReal hut = -A * PetscSinReal(t) * cx * sy;
  PetscReal hvt = A * PetscCosReal(t) * sx * cy;
  PetscReal hx  = A * PetscSinReal(t) * kx * cx * sy;
  PetscReal hy  = A * PetscSinReal(t) * ky * sx * cy;
  PetscReal hux = -A * PetscCosReal(t) * kx * sx * sy;
  PetscReal huy = A * PetscCosReal(t) * ky * cx * cy;
  PetscReal hvx = A * PetscSinReal(t) * kx * cx * cy;
  PetscReal hvy = -A * PetscSinReal(t) * ky * sx * sy;
  PetscReal hu2_over_h_x, huhv_over_h_y, huhv_over_h_x, hv2_over_h_y;

  ManufacturedSolution2D(Lx, Ly, x, y, t, h0, A, &h, &hu, &hv);
  hu2_over_h_x  = (2.0 * hu * hux * h - hu * hu * hx) / (h * h);
  huhv_over_h_y = ((huy * hv + hu * hvy) * h - hu * hv * hy) / (h * h);
  huhv_over_h_x = ((hux * hv + hu * hvx) * h - hu * hv * hx) / (h * h);
  hv2_over_h_y  = (2.0 * hv * hvy * h - hv * hv * hy) / (h * h);

  *S_h  = ht + hux + hvy;
  *S_hu = hut + hu2_over_h_x + g * h * hx + huhv_over_h_y;
  *S_hv = hvt + huhv_over_h_x + hv2_over_h_y + g * h * hy;
}

static inline void ComputeFluxX(PetscReal g, PetscReal h, PetscReal hu, PetscReal hv, PetscReal *F_h, PetscReal *F_hu, PetscReal *F_hv, PetscReal *u, PetscReal *c)
{
  if (h > EX4_DRY_TOL) {
    *u    = hu / h;
    *c    = PetscSqrtReal(g * h);
    *F_h  = hu;
    *F_hu = hu * *u + 0.5 * g * h * h;
    *F_hv = hu * (hv / h);
  } else {
    *u    = 0.0;
    *c    = 0.0;
    *F_h  = 0.0;
    *F_hu = 0.0;
    *F_hv = 0.0;
  }
}

static inline void ComputeFluxY(PetscReal g, PetscReal h, PetscReal hu, PetscReal hv, PetscReal *G_h, PetscReal *G_hu, PetscReal *G_hv, PetscReal *v, PetscReal *c)
{
  if (h > EX4_DRY_TOL) {
    *v    = hv / h;
    *c    = PetscSqrtReal(g * h);
    *G_h  = hv;
    *G_hu = hv * (hu / h);
    *G_hv = hv * *v + 0.5 * g * h * h;
  } else {
    *v    = 0.0;
    *c    = 0.0;
    *G_h  = 0.0;
    *G_hu = 0.0;
    *G_hv = 0.0;
  }
}

static inline PetscErrorCode ShallowWaterRHS2D(TS ts, PetscReal t, Vec X, Vec F_vec, PetscCtx ctx)
{
  ShallowWater2DCtx   *sw = (ShallowWater2DCtx *)ctx;
  Vec                  X_local;
  const PetscScalar ***x;
  PetscScalar       ***f;
  PetscInt             xs, ys, xm, ym, i, j;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(sw->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMGetLocalVector(sw->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecGetArrayDOFWrite(sw->da, F_vec, &f));

  /* Only the Rusanov flux is implemented. */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal h      = PetscRealPart(x[j][i][0]);
      PetscReal hu     = PetscRealPart(x[j][i][1]);
      PetscReal hv     = PetscRealPart(x[j][i][2]);
      PetscReal h_im1  = PetscRealPart(x[j][i - 1][0]);
      PetscReal hu_im1 = PetscRealPart(x[j][i - 1][1]);
      PetscReal hv_im1 = PetscRealPart(x[j][i - 1][2]);
      PetscReal h_ip1  = PetscRealPart(x[j][i + 1][0]);
      PetscReal hu_ip1 = PetscRealPart(x[j][i + 1][1]);
      PetscReal hv_ip1 = PetscRealPart(x[j][i + 1][2]);
      PetscReal h_jm1  = PetscRealPart(x[j - 1][i][0]);
      PetscReal hu_jm1 = PetscRealPart(x[j - 1][i][1]);
      PetscReal hv_jm1 = PetscRealPart(x[j - 1][i][2]);
      PetscReal h_jp1  = PetscRealPart(x[j + 1][i][0]);
      PetscReal hu_jp1 = PetscRealPart(x[j + 1][i][1]);
      PetscReal hv_jp1 = PetscRealPart(x[j + 1][i][2]);
      PetscReal F_h_i, F_hu_i, F_hv_i, u, c;
      PetscReal F_h_im1, F_hu_im1, F_hv_im1, u_im1, c_im1;
      PetscReal F_h_ip1, F_hu_ip1, F_hv_ip1, u_ip1, c_ip1;
      PetscReal G_h_j, G_hu_j, G_hv_j, v, c_y;
      PetscReal G_h_jm1, G_hu_jm1, G_hv_jm1, v_jm1, c_jm1;
      PetscReal G_h_jp1, G_hu_jp1, G_hv_jp1, v_jp1, c_jp1;
      PetscReal alpha_left, alpha_right, beta_bottom, beta_top;
      PetscReal flux_h_left, flux_hu_left, flux_hv_left;
      PetscReal flux_h_right, flux_hu_right, flux_hv_right;
      PetscReal flux_h_bottom, flux_hu_bottom, flux_hv_bottom;
      PetscReal flux_h_top, flux_hu_top, flux_hv_top;

      ComputeFluxX(sw->g, h, hu, hv, &F_h_i, &F_hu_i, &F_hv_i, &u, &c);
      ComputeFluxX(sw->g, h_im1, hu_im1, hv_im1, &F_h_im1, &F_hu_im1, &F_hv_im1, &u_im1, &c_im1);
      ComputeFluxX(sw->g, h_ip1, hu_ip1, hv_ip1, &F_h_ip1, &F_hu_ip1, &F_hv_ip1, &u_ip1, &c_ip1);
      ComputeFluxY(sw->g, h, hu, hv, &G_h_j, &G_hu_j, &G_hv_j, &v, &c_y);
      ComputeFluxY(sw->g, h_jm1, hu_jm1, hv_jm1, &G_h_jm1, &G_hu_jm1, &G_hv_jm1, &v_jm1, &c_jm1);
      ComputeFluxY(sw->g, h_jp1, hu_jp1, hv_jp1, &G_h_jp1, &G_hu_jp1, &G_hv_jp1, &v_jp1, &c_jp1);

      alpha_left  = PetscMax(PetscAbsReal(u_im1) + c_im1, PetscAbsReal(u) + c);
      alpha_right = PetscMax(PetscAbsReal(u) + c, PetscAbsReal(u_ip1) + c_ip1);
      beta_bottom = PetscMax(PetscAbsReal(v_jm1) + c_jm1, PetscAbsReal(v) + c_y);
      beta_top    = PetscMax(PetscAbsReal(v) + c_y, PetscAbsReal(v_jp1) + c_jp1);

      flux_h_left    = 0.5 * (F_h_im1 + F_h_i - alpha_left * (h - h_im1));
      flux_hu_left   = 0.5 * (F_hu_im1 + F_hu_i - alpha_left * (hu - hu_im1));
      flux_hv_left   = 0.5 * (F_hv_im1 + F_hv_i - alpha_left * (hv - hv_im1));
      flux_h_right   = 0.5 * (F_h_i + F_h_ip1 - alpha_right * (h_ip1 - h));
      flux_hu_right  = 0.5 * (F_hu_i + F_hu_ip1 - alpha_right * (hu_ip1 - hu));
      flux_hv_right  = 0.5 * (F_hv_i + F_hv_ip1 - alpha_right * (hv_ip1 - hv));
      flux_h_bottom  = 0.5 * (G_h_jm1 + G_h_j - beta_bottom * (h - h_jm1));
      flux_hu_bottom = 0.5 * (G_hu_jm1 + G_hu_j - beta_bottom * (hu - hu_jm1));
      flux_hv_bottom = 0.5 * (G_hv_jm1 + G_hv_j - beta_bottom * (hv - hv_jm1));
      flux_h_top     = 0.5 * (G_h_j + G_h_jp1 - beta_top * (h_jp1 - h));
      flux_hu_top    = 0.5 * (G_hu_j + G_hu_jp1 - beta_top * (hu_jp1 - hu));
      flux_hv_top    = 0.5 * (G_hv_j + G_hv_jp1 - beta_top * (hv_jp1 - hv));

      f[j][i][0] = -(flux_h_right - flux_h_left) / sw->dx - (flux_h_top - flux_h_bottom) / sw->dy;
      f[j][i][1] = -(flux_hu_right - flux_hu_left) / sw->dx - (flux_hu_top - flux_hu_bottom) / sw->dy;
      f[j][i][2] = -(flux_hv_right - flux_hv_left) / sw->dx - (flux_hv_top - flux_hv_bottom) / sw->dy;

      if (sw->verify_mms) {
        PetscReal x_coord = ((PetscReal)i + 0.5) * sw->dx;
        PetscReal y_coord = ((PetscReal)j + 0.5) * sw->dy;
        PetscReal S_h = 0.0, S_hu = 0.0, S_hv = 0.0;

        ManufacturedSource2D(sw->Lx, sw->Ly, x_coord, y_coord, t, sw->g, sw->h0, sw->Ax, &S_h, &S_hu, &S_hv);
        f[j][i][0] += S_h;
        f[j][i][1] += S_hu;
        f[j][i][2] += S_hv;
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecRestoreArrayDOFWrite(sw->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(sw->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode ShallowWater2DContextCreate(DM da, PetscInt nx, PetscInt ny, PetscReal Lx, PetscReal Ly, PetscReal g, PetscReal dt, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscBool verify_mms, Ex4FluxType flux_type, ShallowWater2DCtx **ctx)
{
  ShallowWater2DCtx *sw;

  PetscFunctionBeginUser;
  /* ManufacturedSource2D divides by h*h, where h = h0 + A*sin(t)*sx*sy ranges in [h0-A, h0+A].
     Enforce h0 - Ax > EX4_DRY_TOL once on the DA's comm so misuse fails collectively at setup
     instead of deadlocking on a per-cell SETERRQ from one rank during the RHS evaluation. */
  if (verify_mms) PetscCheck(h0 - Ax > EX4_DRY_TOL, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "MMS amplitude Ax (%g) must leave h0 (%g) > EX4_DRY_TOL (%g)", (double)Ax, (double)h0, (double)EX4_DRY_TOL);
  PetscCall(PetscNew(&sw));
  /* Borrowed reference; caller owns the DM and ShallowWater2DContextDestroy() does not free it. */
  sw->da         = da;
  sw->nx         = nx;
  sw->ny         = ny;
  sw->Lx         = Lx;
  sw->Ly         = Ly;
  sw->g          = g;
  sw->dx         = Lx / nx;
  sw->dy         = Ly / ny;
  sw->dt         = dt;
  sw->t_current  = 0.0;
  sw->h0         = h0;
  sw->Ax         = Ax;
  sw->Ay         = Ay;
  sw->verify_mms = verify_mms;
  sw->flux_type  = flux_type;

  PetscCall(TSCreate(PetscObjectComm((PetscObject)da), &sw->ts));
  PetscCall(TSSetProblemType(sw->ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(sw->ts, NULL, ShallowWaterRHS2D, sw));
  PetscCall(TSSetType(sw->ts, TSRK));
  PetscCall(TSRKSetType(sw->ts, TSRK4));
  PetscCall(TSSetTimeStep(sw->ts, dt));
  PetscCall(TSSetMaxSteps(sw->ts, 1));
  PetscCall(TSSetMaxTime(sw->ts, dt));
  PetscCall(TSSetExactFinalTime(sw->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(sw->ts));

  *ctx = sw;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode ShallowWater2DContextDestroy(ShallowWater2DCtx **ctx)
{
  PetscFunctionBeginUser;
  if (!*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Advance a single state vector one TS step starting from physical time t_start. The MMS source
   (when verify_mms is enabled in the context) is evaluated against the TS time, so callers driving
   multi-step verification runs must pass the cumulative start time of each step rather than 0. */
static inline PetscErrorCode ShallowWaterStep2DVec(ShallowWater2DCtx *sw, PetscReal t_start, Vec x)
{
  PetscFunctionBeginUser;
  /* The TSSetTimeStep and TSSetMaxSteps calls below look redundant with ShallowWater2DContextCreate()
     but are load-bearing: TSSetMaxTime forces the last sub-step to shrink to land exactly on
     t_start+dt (and TSAdapt may have shrunk dt during the step), so the trailing dt persists into the
     next call unless reset; and -ts_max_steps from the command line overrides the create-time 1, so
     re-asserting it per call enforces this wrapper's "exactly one step per call" contract. */
  PetscCall(TSSetTime(sw->ts, t_start));
  PetscCall(TSSetStepNumber(sw->ts, 0));
  PetscCall(TSSetTimeStep(sw->ts, sw->dt));
  PetscCall(TSSetMaxSteps(sw->ts, 1));
  PetscCall(TSSetMaxTime(sw->ts, t_start + sw->dt));
  PetscCall(TSSolve(sw->ts, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode ShallowWaterStep2D(Mat ensemble, PetscCtx ctx)
{
  ShallowWater2DCtx *sw = (ShallowWater2DCtx *)ctx;
  PetscInt           n, j;
  PetscBool          isdense;

  PetscFunctionBeginUser;
  PetscCheck(!sw->verify_mms, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_SUP, "MMS verification is not supported for the ensemble Mat path; the per-step time tracked here can drift from external Vec callers");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)ensemble, &isdense, MATSEQDENSE, MATMPIDENSE, MATDENSE, ""));
  PetscCheck(isdense, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_SUP, "ShallowWaterStep2D requires a dense ensemble Mat (got non-dense type)");
  PetscCall(MatGetSize(ensemble, NULL, &n));
  for (j = 0; j < n; j++) {
    Vec col;

    PetscCall(MatDenseGetColumnVecWrite(ensemble, j, &col));
    PetscCall(ShallowWaterStep2DVec(sw, sw->t_current, col));
    PetscCall(MatDenseRestoreColumnVecWrite(ensemble, j, &col));
  }
  sw->t_current += sw->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void ShallowWaterSolution_Wave2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal g, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal *h, PetscReal *hu, PetscReal *hv)
{
  PetscReal c       = PetscSqrtReal(g * h0);
  PetscReal kx      = 2.0 * PETSC_PI / Lx;
  PetscReal ky      = 2.0 * PETSC_PI / Ly;
  PetscReal omega_x = c * kx;
  PetscReal omega_y = c * ky;

  *h  = h0 + Ax * PetscSinReal(kx * x - omega_x * t) + Ay * PetscSinReal(ky * y - omega_y * t);
  *hu = (*h) * (c / h0) * Ax * PetscCosReal(kx * x - omega_x * t);
  *hv = (*h) * (c / h0) * Ay * PetscCosReal(ky * y - omega_y * t);
}

/*
  SetupForwardProblem - Create DM, shallow water context, and solution vector
*/
static inline PetscErrorCode SetupForwardProblem(PetscInt nx, PetscInt ny, PetscReal Lx, PetscReal Ly, PetscReal g, PetscReal dt, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscBool verify_mms, Ex4FluxType flux_type, DM *da_state, ShallowWater2DCtx **sw_ctx, Vec *x)
{
  PetscFunctionBeginUser;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, da_state));
  PetscCall(DMSetFromOptions(*da_state));
  PetscCall(DMSetUp(*da_state));
  PetscCall(ShallowWater2DContextCreate(*da_state, nx, ny, Lx, Ly, g, dt, h0, Ax, Ay, verify_mms, flux_type, sw_ctx));
  PetscCall(DMCreateGlobalVector(*da_state, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SetInitialCondition - Set initial condition on solution vector from analytic solution
*/
static inline PetscErrorCode SetInitialCondition(DM da_state, Vec x, ShallowWater2DCtx *sw, PetscBool use_mms)
{
  PetscScalar ***x_array;
  PetscInt       xs, ys, xm, ym;

  PetscFunctionBeginUser;
  PetscCheck(!use_mms || PetscAbsReal(sw->Ax - sw->Ay) <= 100 * PETSC_MACHINE_EPSILON * PetscMax(PetscAbsReal(sw->Ax), PetscAbsReal(sw->Ay)), PetscObjectComm((PetscObject)da_state), PETSC_ERR_ARG_INCOMP, "MMS requires isotropic amplitude (Ax == Ay); got Ax=%g, Ay=%g",
             (double)sw->Ax, (double)sw->Ay);
  PetscCall(DMDAGetCorners(da_state, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAVecGetArrayDOFWrite(da_state, x, &x_array));
  for (PetscInt j = ys; j < ys + ym; j++) {
    for (PetscInt i = xs; i < xs + xm; i++) {
      PetscReal xc = ((PetscReal)i + 0.5) * sw->dx;
      PetscReal yc = ((PetscReal)j + 0.5) * sw->dy;
      PetscReal h, hu, hv;

      if (use_mms) ManufacturedSolution2D(sw->Lx, sw->Ly, xc, yc, 0.0, sw->h0, sw->Ax, &h, &hu, &hv);
      else ShallowWaterSolution_Wave2D(sw->Lx, sw->Ly, xc, yc, 0.0, sw->g, sw->h0, sw->Ax, sw->Ay, &h, &hu, &hv);
      x_array[j][i][0] = h;
      x_array[j][i][1] = hu;
      x_array[j][i][2] = hv;
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFWrite(da_state, x, &x_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
