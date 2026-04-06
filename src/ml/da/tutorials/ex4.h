#pragma once

#include <petscdmda.h>
#include <petscts.h>

typedef enum {
  EX4_FLUX_RUSANOV,
  EX4_FLUX_MC
} Ex4FluxType;

static const char *const Ex4FluxTypes[] = {"rusanov", "mc", "Ex4FluxType", "EX4_FLUX_", NULL};

typedef struct {
  DM          da;
  PetscInt    nx, ny;
  PetscReal   Lx, Ly;
  PetscReal   dx, dy;
  PetscReal   g;
  PetscReal   dt;
  TS          ts;
  PetscReal   h0;
  PetscReal   Ax, Ay;
  PetscBool   verify_mms;
  Ex4FluxType flux_type;
} ShallowWater2DCtx;

static PetscErrorCode ManufacturedSolution2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal *h, PetscReal *hu, PetscReal *hv)
{
  PetscReal sx, sy, cx, cy;

  PetscFunctionBeginUser;
  sx = PetscSinReal(2.0 * PETSC_PI * x / Lx);
  sy = PetscSinReal(2.0 * PETSC_PI * y / Ly);
  cx = PetscCosReal(2.0 * PETSC_PI * x / Lx);
  cy = PetscCosReal(2.0 * PETSC_PI * y / Ly);

  *h  = h0 + Ax * PetscSinReal(t) * sx * sy;
  *hu = Ay * PetscCosReal(t) * cx * sy;
  *hv = Ay * PetscSinReal(t) * sx * cy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ManufacturedSource2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal g, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal *S_h, PetscReal *S_hu, PetscReal *S_hv)
{
  PetscReal h, hu, hv;
  PetscReal sx, sy, cx, cy;
  PetscReal kx, ky;
  PetscReal ht, hut, hvt;
  PetscReal hx, hy, hux, huy, hvx, hvy;
  PetscReal hu2_over_h_x, huhv_over_h_y, huhv_over_h_x, hv2_over_h_y;

  PetscFunctionBeginUser;
  PetscCall(ManufacturedSolution2D(Lx, Ly, x, y, t, h0, Ax, Ay, &h, &hu, &hv));

  kx = 2.0 * PETSC_PI / Lx;
  ky = 2.0 * PETSC_PI / Ly;
  sx = PetscSinReal(kx * x);
  sy = PetscSinReal(ky * y);
  cx = PetscCosReal(kx * x);
  cy = PetscCosReal(ky * y);

  ht  = Ax * PetscCosReal(t) * sx * sy;
  hut = -Ay * PetscSinReal(t) * cx * sy;
  hvt = Ay * PetscCosReal(t) * sx * cy;

  hx  = Ax * PetscSinReal(t) * kx * cx * sy;
  hy  = Ax * PetscSinReal(t) * ky * sx * cy;
  hux = -Ay * PetscCosReal(t) * kx * sx * sy;
  huy = Ay * PetscCosReal(t) * ky * cx * cy;
  hvx = Ay * PetscSinReal(t) * kx * cx * cy;
  hvy = -Ay * PetscSinReal(t) * ky * sx * sy;

  hu2_over_h_x  = (2.0 * hu * hux * h - hu * hu * hx) / (h * h);
  huhv_over_h_y = ((huy * hv + hu * hvy) * h - hu * hv * hy) / (h * h);
  huhv_over_h_x = ((hux * hv + hu * hvx) * h - hu * hv * hx) / (h * h);
  hv2_over_h_y  = (2.0 * hv * hvy * h - hv * hv * hy) / (h * h);

  *S_h  = ht + hux + hvy;
  *S_hu = hut + hu2_over_h_x + g * h * hx + huhv_over_h_y;
  *S_hv = hvt + huhv_over_h_x + hv2_over_h_y + g * h * hy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ComputeFluxX(PetscReal g, PetscReal h, PetscReal hu, PetscReal hv, PetscReal *F_h, PetscReal *F_hu, PetscReal *F_hv, PetscReal *u, PetscReal *c)
{
  if (h > 1e-10) {
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

static void ComputeFluxY(PetscReal g, PetscReal h, PetscReal hu, PetscReal hv, PetscReal *G_h, PetscReal *G_hu, PetscReal *G_hv, PetscReal *v, PetscReal *c)
{
  if (h > 1e-10) {
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

static PetscErrorCode ShallowWaterRHS2D(TS ts, PetscReal t, Vec X, Vec F_vec, PetscCtx ctx)
{
  ShallowWater2DCtx   *sw = (ShallowWater2DCtx *)ctx;
  Vec                  X_local;
  const PetscScalar ***x;
  PetscScalar       ***f;
  PetscInt             xs, ys, xm, ym, i, j;

  PetscFunctionBeginUser;
  (void)ts;
  PetscCall(DMDAGetCorners(sw->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMGetLocalVector(sw->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecGetArrayDOF(sw->da, F_vec, &f));

  if (sw->flux_type == EX4_FLUX_RUSANOV) {
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

          PetscCall(ManufacturedSource2D(sw->Lx, sw->Ly, x_coord, y_coord, t, sw->g, sw->h0, sw->Ax, sw->Ay, &S_h, &S_hu, &S_hv));
          f[j][i][0] += S_h;
          f[j][i][1] += S_hu;
          f[j][i][2] += S_hv;
        }
      }
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)sw->da), PETSC_ERR_SUP, "MC limiter not yet implemented for 2D");
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecRestoreArrayDOF(sw->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(sw->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ShallowWater2DContextCreate(DM da, PetscInt nx, PetscInt ny, PetscReal Lx, PetscReal Ly, PetscReal g, PetscReal dt, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscBool verify_mms, Ex4FluxType flux_type, ShallowWater2DCtx **ctx)
{
  ShallowWater2DCtx *sw;

  PetscFunctionBeginUser;
  PetscCheck(flux_type != EX4_FLUX_MC, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "MC flux limiter not yet implemented for 2D; use -ex4_flux rusanov");
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

static PetscErrorCode ShallowWater2DContextDestroy(ShallowWater2DCtx **ctx)
{
  PetscFunctionBeginUser;
  if (!ctx || !*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ShallowWaterStep2D(Vec input, Vec output, PetscCtx ctx)
{
  ShallowWater2DCtx *sw = (ShallowWater2DCtx *)ctx;

  PetscFunctionBeginUser;
  if (input != output) PetscCall(VecCopy(input, output));
  PetscCall(TSSetTime(sw->ts, 0.0));
  PetscCall(TSSetStepNumber(sw->ts, 0));
  PetscCall(TSSetTimeStep(sw->ts, sw->dt));
  PetscCall(TSSetMaxSteps(sw->ts, 1));
  PetscCall(TSSetMaxTime(sw->ts, sw->dt));
  PetscCall(TSSolve(sw->ts, output));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ShallowWaterSolution_Wave2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal g, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal *h, PetscReal *hu, PetscReal *hv)
{
  PetscReal kx, ky, omega_x, omega_y, c;

  PetscFunctionBeginUser;
  c       = PetscSqrtReal(g * h0);
  kx      = 2.0 * PETSC_PI / Lx;
  ky      = 2.0 * PETSC_PI / Ly;
  omega_x = c * kx;
  omega_y = c * ky;

  *h  = h0 + Ax * PetscSinReal(kx * x - omega_x * t) + Ay * PetscSinReal(ky * y - omega_y * t);
  *hu = (*h) * (c / h0) * Ax * PetscCosReal(kx * x - omega_x * t);
  *hv = (*h) * (c / h0) * Ay * PetscCosReal(ky * y - omega_y * t);
  PetscFunctionReturn(PETSC_SUCCESS);
}
