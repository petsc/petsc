static char help[] = "2D Shallow water equations with traveling wave verification and LETKF data assimilation.\n"
                     "Implements 2D shallow water equations with 3 DOF per grid point (h, hu, hv).\n\n"
                     "Usage modes:\n"
                     "  1. LETKF data assimilation:\n"
                     "     ./ex4 -steps 100 -nx 41 -ny 41 -petscda_type letkf -ensemble_size 30\n"
                     "  2. Verification mode (compare to analytic solution):\n"
                     "     ./ex4 -verification_mode -steps 500 -nx 80 -ny 80 -verification_freq 50\n"
                     "  3. Combined mode:\n"
                     "     ./ex4 -steps 100 -ensemble_size 20 -petscda_type letkf -enable_verification\n\n";

#include <petscda.h>
#include <petscdmda.h>
#include <petscts.h>

/* Default parameter values */
#define DEFAULT_NX            40
#define DEFAULT_NY            40
#define DEFAULT_STEPS         100
#define DEFAULT_OBS_FREQ      5
#define DEFAULT_RANDOM_SEED   12345
#define DEFAULT_G             9.81
#define DEFAULT_DT            0.02
#define DEFAULT_LX            80.0
#define DEFAULT_LY            80.0
#define DEFAULT_H0            1.5
#define DEFAULT_AX            0.2
#define DEFAULT_AY            0.2
#define DEFAULT_OBS_ERROR_STD 0.01
#define DEFAULT_ENSEMBLE_SIZE 30
#define DEFAULT_PROGRESS_FREQ 10
#define DEFAULT_OBS_STRIDE    2
#define SPINUP_STEPS          0

/* Minimum valid parameter values */
#define MIN_ENSEMBLE_SIZE 2
#define MIN_OBS_FREQ      1

/* Flux scheme types */
typedef enum {
  EX4_FLUX_RUSANOV,
  EX4_FLUX_MC
} Ex4FluxType;

static const char *const Ex4FluxTypes[] = {"rusanov", "mc", "Ex4FluxType", "EX4_FLUX_", NULL};

typedef struct {
  DM          da;        /* 2D periodic DMDA for state */
  PetscInt    nx, ny;    /* Grid dimensions */
  PetscReal   Lx, Ly;    /* Domain size */
  PetscReal   dx, dy;    /* Grid spacing */
  PetscReal   g;         /* Gravity */
  PetscReal   dt;        /* Time step */
  TS          ts;        /* Time stepper */
  PetscReal   h0;        /* Mean height */
  PetscReal   Ax, Ay;    /* Wave amplitudes */
  Ex4FluxType flux_type; /* Flux scheme */
} ShallowWater2DCtx;

/*
  ComputeFluxX - Compute physical flux in x-direction for shallow water
*/
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

/*
  ComputeFluxY - Compute physical flux in y-direction for shallow water
*/
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

/*
  ShallowWaterRHS2D - Compute the right-hand side of the 2D shallow water equations
*/
static PetscErrorCode ShallowWaterRHS2D(TS ts, PetscReal t, Vec X, Vec F_vec, PetscCtx ctx)
{
  ShallowWater2DCtx   *sw = (ShallowWater2DCtx *)ctx;
  Vec                  X_local;
  const PetscScalar ***x;
  PetscScalar       ***f;
  PetscInt             xs, ys, xm, ym, i, j;

  PetscFunctionBeginUser;
  (void)ts;
  (void)t;

  PetscCall(DMDAGetCorners(sw->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMGetLocalVector(sw->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecGetArrayDOF(sw->da, F_vec, &f));

  if (sw->flux_type == EX4_FLUX_RUSANOV) {
    /* First-order Rusanov (Local Lax-Friedrichs) scheme */
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

        /* X-direction fluxes */
        PetscReal F_h_i, F_hu_i, F_hv_i, u, c;
        PetscReal F_h_im1, F_hu_im1, F_hv_im1, u_im1, c_im1;
        PetscReal F_h_ip1, F_hu_ip1, F_hv_ip1, u_ip1, c_ip1;

        ComputeFluxX(sw->g, h, hu, hv, &F_h_i, &F_hu_i, &F_hv_i, &u, &c);
        ComputeFluxX(sw->g, h_im1, hu_im1, hv_im1, &F_h_im1, &F_hu_im1, &F_hv_im1, &u_im1, &c_im1);
        ComputeFluxX(sw->g, h_ip1, hu_ip1, hv_ip1, &F_h_ip1, &F_hu_ip1, &F_hv_ip1, &u_ip1, &c_ip1);

        PetscReal alpha_left  = PetscMax(PetscAbsReal(u_im1) + c_im1, PetscAbsReal(u) + c);
        PetscReal alpha_right = PetscMax(PetscAbsReal(u) + c, PetscAbsReal(u_ip1) + c_ip1);

        PetscReal flux_h_left  = 0.5 * (F_h_im1 + F_h_i - alpha_left * (h - h_im1));
        PetscReal flux_hu_left = 0.5 * (F_hu_im1 + F_hu_i - alpha_left * (hu - hu_im1));
        PetscReal flux_hv_left = 0.5 * (F_hv_im1 + F_hv_i - alpha_left * (hv - hv_im1));

        PetscReal flux_h_right  = 0.5 * (F_h_i + F_h_ip1 - alpha_right * (h_ip1 - h));
        PetscReal flux_hu_right = 0.5 * (F_hu_i + F_hu_ip1 - alpha_right * (hu_ip1 - hu));
        PetscReal flux_hv_right = 0.5 * (F_hv_i + F_hv_ip1 - alpha_right * (hv_ip1 - hv));

        /* Y-direction fluxes */
        PetscReal G_h_j, G_hu_j, G_hv_j, v, c_y;
        PetscReal G_h_jm1, G_hu_jm1, G_hv_jm1, v_jm1, c_jm1;
        PetscReal G_h_jp1, G_hu_jp1, G_hv_jp1, v_jp1, c_jp1;

        ComputeFluxY(sw->g, h, hu, hv, &G_h_j, &G_hu_j, &G_hv_j, &v, &c_y);
        ComputeFluxY(sw->g, h_jm1, hu_jm1, hv_jm1, &G_h_jm1, &G_hu_jm1, &G_hv_jm1, &v_jm1, &c_jm1);
        ComputeFluxY(sw->g, h_jp1, hu_jp1, hv_jp1, &G_h_jp1, &G_hu_jp1, &G_hv_jp1, &v_jp1, &c_jp1);

        PetscReal beta_bottom = PetscMax(PetscAbsReal(v_jm1) + c_jm1, PetscAbsReal(v) + c_y);
        PetscReal beta_top    = PetscMax(PetscAbsReal(v) + c_y, PetscAbsReal(v_jp1) + c_jp1);

        PetscReal flux_h_bottom  = 0.5 * (G_h_jm1 + G_h_j - beta_bottom * (h - h_jm1));
        PetscReal flux_hu_bottom = 0.5 * (G_hu_jm1 + G_hu_j - beta_bottom * (hu - hu_jm1));
        PetscReal flux_hv_bottom = 0.5 * (G_hv_jm1 + G_hv_j - beta_bottom * (hv - hv_jm1));

        PetscReal flux_h_top  = 0.5 * (G_h_j + G_h_jp1 - beta_top * (h_jp1 - h));
        PetscReal flux_hu_top = 0.5 * (G_hu_j + G_hu_jp1 - beta_top * (hu_jp1 - hu));
        PetscReal flux_hv_top = 0.5 * (G_hv_j + G_hv_jp1 - beta_top * (hv_jp1 - hv));

        /* Update RHS using finite volume method */
        f[j][i][0] = -(flux_h_right - flux_h_left) / sw->dx - (flux_h_top - flux_h_bottom) / sw->dy;
        f[j][i][1] = -(flux_hu_right - flux_hu_left) / sw->dx - (flux_hu_top - flux_hu_bottom) / sw->dy;
        f[j][i][2] = -(flux_hv_right - flux_hv_left) / sw->dx - (flux_hv_top - flux_hv_bottom) / sw->dy;
      }
    }
  } else {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MC limiter not yet implemented for 2D");
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(sw->da, X_local, (void *)&x));
  PetscCall(DMDAVecRestoreArrayDOF(sw->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(sw->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWater2DContextCreate - Create and initialize a 2D shallow water context
*/
static PetscErrorCode ShallowWater2DContextCreate(DM da, PetscInt nx, PetscInt ny, PetscReal Lx, PetscReal Ly, PetscReal g, PetscReal dt, PetscReal h0, PetscReal Ax, PetscReal Ay, Ex4FluxType flux_type, ShallowWater2DCtx **ctx)
{
  ShallowWater2DCtx *sw;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sw));
  sw->da        = da;
  sw->nx        = nx;
  sw->ny        = ny;
  sw->Lx        = Lx;
  sw->Ly        = Ly;
  sw->g         = g;
  sw->dx        = Lx / nx;
  sw->dy        = Ly / ny;
  sw->dt        = dt;
  sw->h0        = h0;
  sw->Ax        = Ax;
  sw->Ay        = Ay;
  sw->flux_type = flux_type;

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

/*
  ShallowWater2DContextDestroy - Destroy a 2D shallow water context
*/
static PetscErrorCode ShallowWater2DContextDestroy(ShallowWater2DCtx **ctx)
{
  PetscFunctionBeginUser;
  if (!ctx || !*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterStep2D - Advance state vector one time step
*/
static PetscErrorCode ShallowWaterStep2D(Vec input, Vec output, PetscCtx ctx)
{
  ShallowWater2DCtx *sw = (ShallowWater2DCtx *)ctx;

  PetscFunctionBeginUser;
  if (input != output) PetscCall(VecCopy(input, output));

  PetscCall(TSSetTime(sw->ts, 0.0));
  PetscCall(TSSetStepNumber(sw->ts, 0));
  PetscCall(TSSetMaxTime(sw->ts, sw->dt));
  PetscCall(TSSolve(sw->ts, output));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterSolution_Wave2D - Analytic 2D traveling wave solution
*/
static PetscErrorCode ShallowWaterSolution_Wave2D(PetscReal Lx, PetscReal Ly, PetscReal x, PetscReal y, PetscReal t, PetscReal g, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal *h, PetscReal *hu, PetscReal *hv)
{
  PetscReal kx, ky, omega_x, omega_y, c;

  PetscFunctionBeginUser;
  /* Wave parameters */
  c       = PetscSqrtReal(g * h0);
  kx      = 2.0 * PETSC_PI / Lx;
  ky      = 2.0 * PETSC_PI / Ly;
  omega_x = c * kx;
  omega_y = c * ky;

  /* Height field: superposition of waves in x and y */
  PetscReal h_pert_x = Ax * PetscSinReal(kx * x - omega_x * t);
  PetscReal h_pert_y = Ay * PetscSinReal(ky * y - omega_y * t);
  *h                 = h0 + h_pert_x + h_pert_y;

  /* Velocity fields (linearized) */
  PetscReal u = (c / h0) * Ax * PetscCosReal(kx * x - omega_x * t);
  PetscReal v = (c / h0) * Ay * PetscCosReal(ky * y - omega_y * t);

  *hu = (*h) * u;
  *hv = (*h) * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeAnalyticError - Compute error between numerical solution and analytic traveling wave

  Computes L1, L2, and Linf norms of the error in the height field
*/
static PetscErrorCode ComputeAnalyticError(Vec numerical, DM da, PetscReal time, PetscReal Lx, PetscReal Ly, PetscReal g, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal *L1_error, PetscReal *L2_error, PetscReal *Linf_error)
{
  const PetscScalar ***x_num;
  PetscInt             xs, ys, xm, ym, i, j;
  PetscReal            dx, dy, dA;
  PetscReal            L1_local = 0.0, L2_local = 0.0, Linf_local = 0.0;
  PetscInt             nx, ny;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  dx = Lx / nx;
  dy = Ly / ny;
  dA = dx * dy;

  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAVecGetArrayDOFRead(da, numerical, (void *)&x_num));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal x = ((PetscReal)i + 0.5) * dx;
      PetscReal y = ((PetscReal)j + 0.5) * dy;
      PetscReal h_exact, hu_exact, hv_exact;
      PetscReal h_num = PetscRealPart(x_num[j][i][0]);

      /* Compute analytic solution at this point */
      PetscCall(ShallowWaterSolution_Wave2D(Lx, Ly, x, y, time, g, h0, Ax, Ay, &h_exact, &hu_exact, &hv_exact));

      /* Compute pointwise error */
      PetscReal error = PetscAbsReal(h_num - h_exact);

      /* Accumulate norms */
      L1_local += error * dA;
      L2_local += error * error * dA;
      Linf_local = PetscMax(Linf_local, error);
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da, numerical, (void *)&x_num));

  /* Global reduction for L1 and L2 norms */
  PetscCallMPI(MPIU_Allreduce(&L1_local, L1_error, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)da)));
  PetscCallMPI(MPIU_Allreduce(&L2_local, L2_error, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)da)));
  *L2_error = PetscSqrtReal(*L2_error);

  /* Global reduction for Linf norm */
  PetscCallMPI(MPIU_Allreduce(&Linf_local, Linf_error, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)da)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateObservationMatrix2D - Create observation matrix H for 2D shallow water

  Observes water height (h) at every obs_stride-th grid point in both x and y directions.
*/
static PetscErrorCode CreateObservationMatrix2D(PetscInt nx, PetscInt ny, PetscInt ndof, PetscInt obs_stride, Vec state, Mat *H, Mat *H1, PetscInt *nobs_out)
{
  PetscInt i, j, obs_idx, local_state_size;
  PetscInt nobs_x, nobs_y, nobs;
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  /* Calculate number of observations */
  nobs_x = (nx + obs_stride - 1) / obs_stride;
  nobs_y = (ny + obs_stride - 1) / obs_stride;
  nobs   = nobs_x * nobs_y;

  PetscCall(VecGetLocalSize(state, &local_state_size));

  /* Create observation matrix H (nobs x nx*ny*ndof) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, local_state_size, nobs, nx * ny * ndof, 1, NULL, 1, NULL, H));
  PetscCall(MatSetFromOptions(*H));

  /* Create H1 for scalar field (nobs x nx*ny) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, local_state_size / ndof, nobs, nx * ny, 1, NULL, 1, NULL, H1));
  PetscCall(MatSetFromOptions(*H1));

  /* Get row ownership range for local process */
  PetscCall(MatGetOwnershipRange(*H, &rstart, &rend));

  /* Observe water height (h) at sparse grid locations - only set local rows */
  obs_idx = 0;
  for (j = 0; j < ny; j += obs_stride) {
    for (i = 0; i < nx; i += obs_stride) {
      if (obs_idx >= rstart && obs_idx < rend) {
        PetscInt grid_idx = j * nx + i;
        /* H1: select grid point */
        PetscCall(MatSetValue(*H1, obs_idx, grid_idx, 1.0, INSERT_VALUES));
        /* H: select h component (first DOF) at that grid point */
        PetscCall(MatSetValue(*H, obs_idx, grid_idx * ndof, 1.0, INSERT_VALUES));
      }
      obs_idx++;
    }
  }

  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(*H1, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H1, MAT_FINAL_ASSEMBLY));

  PetscCall(MatViewFromOptions(*H1, NULL, "-H_view"));
  *nobs_out = nobs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateLocalizationMatrix2D - Create localization matrix Q for 2D shallow water
*/
static PetscErrorCode CreateLocalizationMatrix2D(PetscInt nx, PetscInt ny, PetscInt nobs, Mat *Q)
{
  PetscInt i, j;

  PetscFunctionBeginUser;
  /* Create Q matrix (nx*ny x nobs) - global/no localization for simplicity */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nx * ny, nobs, nobs, NULL, 0, NULL, Q));
  PetscCall(MatSetFromOptions(*Q));

  /* Initialize with no localization: each state variable uses all observations */
  for (i = 0; i < nx * ny; i++) {
    for (j = 0; j < nobs; j++) PetscCall(MatSetValue(*Q, i, j, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeRMSE - Compute root mean square error between two vectors
*/
static PetscErrorCode ComputeRMSE(Vec v1, Vec v2, Vec work, PetscInt n, PetscReal *rmse)
{
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(work, -1.0, v2, v1));
  PetscCall(VecNorm(work, NORM_2, &norm));
  *rmse = norm / PetscSqrtReal((PetscReal)n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ValidateParameters - Validate input parameters
*/
static PetscErrorCode ValidateParameters(PetscInt *nx, PetscInt *ny, PetscInt *nobs, PetscInt *steps, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *g, PetscReal *obs_error_std)
{
  PetscFunctionBeginUser;
  PetscCheck(*nx > 0 && *ny > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid dimensions must be positive");
  PetscCheck(*steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative");
  PetscCheck(*ensemble_size >= MIN_ENSEMBLE_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least %d", MIN_ENSEMBLE_SIZE);

  if (*obs_freq < MIN_OBS_FREQ) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency adjusted from %" PetscInt_FMT " to %d\n", *obs_freq, MIN_OBS_FREQ));
    *obs_freq = MIN_OBS_FREQ;
  }
  if (*obs_freq > *steps && *steps > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency > total steps, no observations will be assimilated.\n"));

  PetscCheck(*dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step must be positive");
  PetscCheck(*obs_error_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observation error std must be positive");
  PetscCheck(PetscIsNormalReal(*g), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Gravitational constant must be a normal real number");
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  /* Configuration parameters */
  const PetscInt ndof                    = 3; /* h, hu, hv */
  PetscInt       nx                      = DEFAULT_NX;
  PetscInt       ny                      = DEFAULT_NY;
  PetscInt       steps                   = DEFAULT_STEPS;
  PetscInt       obs_freq                = DEFAULT_OBS_FREQ;
  PetscInt       random_seed             = DEFAULT_RANDOM_SEED;
  PetscInt       ensemble_size           = DEFAULT_ENSEMBLE_SIZE;
  PetscInt       n_spin                  = SPINUP_STEPS;
  PetscInt       progress_freq           = DEFAULT_PROGRESS_FREQ;
  PetscInt       obs_stride              = DEFAULT_OBS_STRIDE;
  PetscReal      g                       = DEFAULT_G;
  PetscReal      dt                      = DEFAULT_DT;
  PetscReal      Lx                      = DEFAULT_LX;
  PetscReal      Ly                      = DEFAULT_LY;
  PetscReal      h0                      = DEFAULT_H0;
  PetscReal      Ax                      = DEFAULT_AX;
  PetscReal      Ay                      = DEFAULT_AY;
  PetscReal      obs_error_std           = DEFAULT_OBS_ERROR_STD;
  PetscBool      use_fake_localization   = PETSC_FALSE;
  PetscInt       num_observations_vertex = 7;
  Ex4FluxType    flux_type               = EX4_FLUX_RUSANOV;
  char           output_file[PETSC_MAX_PATH_LEN];
  PetscBool      output_enabled      = PETSC_FALSE;
  PetscBool      verification_mode   = PETSC_FALSE;
  PetscBool      enable_verification = PETSC_FALSE;
  PetscInt       verification_freq   = 10;
  FILE          *fp                  = NULL;

  /* PETSc objects */
  ShallowWater2DCtx *sw_ctx = NULL;
  DM                 da_state;
  PetscDA            da;
  Vec                x0, x_mean, x_forecast;
  Vec                truth_state, rmse_work;
  Vec                observation, obs_noise, obs_error_var;
  PetscRandom        rng;
  Mat                Q = NULL, H = NULL, H1 = NULL;
  PetscInt           nobs;

  /* Statistics tracking */
  PetscReal rmse_forecast = 0.0, rmse_analysis = 0.0;
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt  n_stat_steps = 0;
  PetscInt  obs_count    = 0;
  PetscInt  step;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "2D Shallow Water LETKF Example", NULL);
  PetscCall(PetscOptionsInt("-nx", "Number of grid points in x", "", nx, &nx, NULL));
  PetscCall(PetscOptionsInt("-ny", "Number of grid points in y", "", ny, &ny, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsInt("-obs_freq", "Observation frequency", "", obs_freq, &obs_freq, NULL));
  PetscCall(PetscOptionsInt("-obs_stride", "Observation stride (sample every Nth grid point)", "", obs_stride, &obs_stride, NULL));
  PetscCall(PetscOptionsReal("-g", "Gravitational constant", "", g, &g, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-Lx", "Domain length in x", "", Lx, &Lx, NULL));
  PetscCall(PetscOptionsReal("-Ly", "Domain length in y", "", Ly, &Ly, NULL));
  PetscCall(PetscOptionsReal("-h0", "Mean water height", "", h0, &h0, NULL));
  PetscCall(PetscOptionsReal("-Ax", "Wave amplitude in x", "", Ax, &Ax, NULL));
  PetscCall(PetscOptionsReal("-Ay", "Wave amplitude in y", "", Ay, &Ay, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation error standard deviation", "", obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble perturbations", "", random_seed, &random_seed, NULL));
  PetscCall(PetscOptionsInt("-progress_freq", "Print progress every N steps (0 = only first/last)", "", progress_freq, &progress_freq, NULL));
  PetscCall(PetscOptionsString("-output_file", "Output file for visualization data", "", "", output_file, sizeof(output_file), &output_enabled));
  PetscCall(PetscOptionsEnum("-ex4_flux", "Flux scheme (rusanov/mc)", "", Ex4FluxTypes, (PetscEnum)flux_type, (PetscEnum *)&flux_type, NULL));
  PetscCall(PetscOptionsBool("-use_fake_localization", "Use fake localization matrix", "", use_fake_localization, &use_fake_localization, NULL));
  if (!use_fake_localization) PetscCall(PetscOptionsInt("-petscda_letkf_obs_per_vertex", "Number of observations per vertex", "", num_observations_vertex, &num_observations_vertex, NULL));
  PetscCall(PetscOptionsBool("-verification_mode", "Run in pure verification mode (no DA)", "", verification_mode, &verification_mode, NULL));
  PetscCall(PetscOptionsBool("-enable_verification", "Enable verification alongside DA", "", enable_verification, &enable_verification, NULL));
  PetscCall(PetscOptionsInt("-verification_freq", "Frequency for verification error output", "", verification_freq, &verification_freq, NULL));
  PetscOptionsEnd();

  /* Handle verification mode settings */
  if (verification_mode) {
    /* In pure verification mode, disable all DA components */
    enable_verification = PETSC_TRUE;
    ensemble_size       = 0; /* Will skip DA setup */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Running in pure verification mode (no data assimilation)\n"));
  }

  /* Calculate number of observations */
  nobs = ((nx + obs_stride - 1) / obs_stride) * ((ny + obs_stride - 1) / obs_stride);

  /* Set num_observations_vertex for fake localization after nobs is calculated */
  if (use_fake_localization) num_observations_vertex = nobs;

  /* Validate parameters - skip ensemble size check in verification mode */
  if (!verification_mode) {
    PetscCall(ValidateParameters(&nx, &ny, &nobs, &steps, &obs_freq, &ensemble_size, &dt, &g, &obs_error_std));
  } else {
    PetscCheck(nx > 0 && ny > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid dimensions must be positive");
    PetscCheck(steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative");
    PetscCheck(dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step must be positive");
    PetscCheck(PetscIsNormalReal(g), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Gravitational constant must be a normal real number");
  }

  /* Create 2D periodic DMDA with 3 DOF (h, hu, hv) */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, ndof, 2, NULL, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  /* Create shallow water context */
  PetscCall(ShallowWater2DContextCreate(da_state, nx, ny, Lx, Ly, g, dt, h0, Ax, Ay, flux_type, &sw_ctx));

  /* Initialize random number generator */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(PetscRandomSetSeed(rng, (unsigned long)(random_seed + rank)));
  }
  PetscCall(PetscRandomSetFromOptions(rng));
  PetscCall(PetscRandomSeed(rng));

  /* Initialize state vectors */
  PetscCall(DMCreateGlobalVector(da_state, &x0));

  /* Set initial condition from analytic solution */
  {
    PetscScalar ***x_array;
    PetscInt       xs, ys, xm, ym, i, j;
    PetscCall(DMDAGetCorners(da_state, &xs, &ys, NULL, &xm, &ym, NULL));
    PetscCall(DMDAVecGetArrayDOF(da_state, x0, &x_array));
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        PetscReal x = ((PetscReal)i + 0.5) * sw_ctx->dx;
        PetscReal y = ((PetscReal)j + 0.5) * sw_ctx->dy;
        PetscReal h, hu, hv;
        PetscCall(ShallowWaterSolution_Wave2D(Lx, Ly, x, y, 0.0, g, h0, Ax, Ay, &h, &hu, &hv));
        x_array[j][i][0] = h;
        x_array[j][i][1] = hu;
        x_array[j][i][2] = hv;
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(da_state, x0, &x_array));
  }

  /* Initialize truth trajectory */
  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecCopy(x0, truth_state));
  PetscCall(VecDuplicate(x0, &rmse_work));

  /* Spinup if needed */
  if (n_spin > 0) {
    PetscInt spinup_progress_interval = (n_spin >= 10) ? (n_spin / 10) : 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinning up truth trajectory for %" PetscInt_FMT " steps...\n", n_spin));
    for (PetscInt k = 0; k < n_spin; k++) {
      PetscCall(ShallowWaterStep2D(truth_state, truth_state, sw_ctx));
      if ((k + 1) % spinup_progress_interval == 0 || (k + 1) == n_spin) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Spinup progress: %" PetscInt_FMT "/%" PetscInt_FMT "\n", k + 1, n_spin));
    }
    PetscCall(VecCopy(truth_state, x0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinup complete.\n\n"));
  }

  /* Setup data assimilation (skip in pure verification mode) */
  if (!verification_mode) {
    /* Create observation matrix H */
    PetscCall(CreateObservationMatrix2D(nx, ny, ndof, obs_stride, x0, &H, &H1, &nobs));

    /* Initialize observation vectors */
    PetscCall(MatCreateVecs(H, NULL, &observation));
    PetscCall(VecDuplicate(observation, &obs_noise));
    PetscCall(VecDuplicate(observation, &obs_error_var));
    PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

    /* Create and configure PetscDA for ensemble data assimilation */
    PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
    PetscCall(PetscDASetSizes(da, nx * ny * ndof, nobs));
    PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));
    {
      PetscInt local_state_size, local_obs_size;
      PetscCall(VecGetLocalSize(x0, &local_state_size));
      PetscCall(VecGetLocalSize(observation, &local_obs_size));
      PetscCall(PetscDASetLocalSizes(da, local_state_size, local_obs_size));
    }
    PetscCall(PetscDASetNDOF(da, ndof));
    PetscCall(PetscDASetFromOptions(da));
    PetscCall(PetscDAEnsembleGetSize(da, &ensemble_size));
    PetscCall(PetscDASetUp(da));

    /* Initialize ensemble statistics vectors */
    PetscCall(VecDuplicate(x0, &x_mean));
    PetscCall(VecDuplicate(x0, &x_forecast));

    /* Set observation error variance */
    PetscCall(PetscDASetObsErrorVariance(da, obs_error_var));

    /* Create and set localization matrix Q */
    {
      PetscBool isletkf;
      PetscCall(PetscObjectTypeCompare((PetscObject)da, PETSCDALETKF, &isletkf));

      if (!use_fake_localization && isletkf) {
        /* Use PetscDALETKFGetLocalizationMatrix for proper distance-based localization */
        Vec       Vecxyz[3] = {NULL, NULL, NULL};
        Vec       coord;
        DM        cda;
        PetscReal bd[3] = {Lx, Ly, 0};
        PetscInt  cdof;

        /* Set up coordinates for DMDA */
        PetscCall(DMDASetUniformCoordinates(da_state, 0.0, Lx, 0.0, Ly, 0.0, 0.0));

        /* Get coordinate DM and coordinates */
        PetscCall(DMGetCoordinateDM(da_state, &cda));
        PetscCall(DMGetCoordinates(da_state, &coord));
        PetscCall(DMGetBlockSize(cda, &cdof));

        /* Extract x and y coordinates into separate vectors for each grid point */
        /* Need vectors sized for nx*ny points (not nx*ny*ndof)  */
        PetscInt xs, ys, xm, ym;
        PetscCall(DMDAGetCorners(da_state, &xs, &ys, NULL, &xm, &ym, NULL));

        for (PetscInt d = 0; d < 2; d++) {
          PetscScalar ***x_coord_3d;
          PetscScalar   *vec_array;
          PetscInt       i, j, idx;
          PetscInt       local_grid_points = xm * ym;

          /* Create vector for this coordinate component - size should be nx*ny */
          PetscCall(VecCreate(PETSC_COMM_WORLD, &Vecxyz[d]));
          PetscCall(VecSetSizes(Vecxyz[d], local_grid_points, nx * ny));
          PetscCall(VecSetFromOptions(Vecxyz[d]));
          PetscCall(PetscObjectSetName((PetscObject)Vecxyz[d], d == 0 ? "x_coordinate" : "y_coordinate"));

          /* Get coordinate array - it's structured as [x,y] pairs */
          PetscCall(DMDAVecGetArrayDOFRead(cda, coord, (void *)&x_coord_3d));
          PetscCall(VecGetArray(Vecxyz[d], &vec_array));

          /* Copy coordinates from 2D array */
          idx = 0;
          for (j = ys; j < ys + ym; j++) {
            for (i = xs; i < xs + xm; i++) vec_array[idx++] = x_coord_3d[j][i][d];
          }

          PetscCall(VecRestoreArray(Vecxyz[d], &vec_array));
          PetscCall(DMDAVecRestoreArrayDOFRead(cda, coord, (void *)&x_coord_3d));
        }

        /* Get localization matrix using distance-based method */
        PetscCall(PetscDALETKFGetLocalizationMatrix(num_observations_vertex, 1, Vecxyz, bd, H1, &Q));
        PetscCall(VecDestroy(&Vecxyz[0]));
        PetscCall(VecDestroy(&Vecxyz[1]));
        PetscCall(PetscDALETKFSetObsPerVertex(da, num_observations_vertex));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using distance-based localization with %" PetscInt_FMT " observations per vertex\n", num_observations_vertex));
      } else {
        PetscCall(CreateLocalizationMatrix2D(nx, ny, nobs, &Q));
        if (isletkf) {
          PetscCall(PetscDALETKFSetObsPerVertex(da, num_observations_vertex));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using global localization (all observations)\n"));
        }
      }
      PetscCall(PetscDALETKFSetLocalization(da, Q, H));
      PetscCall(MatViewFromOptions(Q, NULL, "-Q_view"));
      PetscCall(MatDestroy(&Q));
    }

    /* Initialize ensemble members with perturbations */
    PetscCall(PetscDAEnsembleInitialize(da, x0, obs_error_std, rng));

    PetscCall(PetscDAViewFromOptions(da, NULL, "-petscda_view"));
  }

  /* Print configuration summary */
  {
    const char *flux_name = (flux_type == EX4_FLUX_RUSANOV) ? "Rusanov (1st order)" : "MC (2nd order)";
    const char *mode_name = verification_mode ? "Verification" : "LETKF";
    PetscReal   dx        = Lx / nx;
    PetscReal   dy        = Ly / ny;
    PetscReal   c         = PetscSqrtReal(g * h0);
    PetscReal   cfl       = dt * c * (1.0 / dx + 1.0 / dy);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D Shallow Water %s Example\n", mode_name));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==============================\n"));
    if (verification_mode) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "  Mode                  : Verification (traveling wave test)\n"
                            "  Flux scheme           : %s\n"
                            "  Grid dimensions       : %" PetscInt_FMT " x %" PetscInt_FMT "\n"
                            "  Domain size           : %.2f x %.2f\n"
                            "  Grid spacing          : dx=%.4f, dy=%.4f\n"
                            "  Mean height (h0)      : %.4f\n"
                            "  Wave amplitudes       : Ax=%.4f, Ay=%.4f\n"
                            "  Gravitational const   : %.4f\n"
                            "  Wave speed (c)        : %.4f\n"
                            "  Time step (dt)        : %.4f\n"
                            "  CFL number            : %.4f\n"
                            "  Total steps           : %" PetscInt_FMT "\n"
                            "  Verification freq     : %" PetscInt_FMT "\n\n",
                            flux_name, nx, ny, (double)Lx, (double)Ly, (double)dx, (double)dy, (double)h0, (double)Ax, (double)Ay, (double)g, (double)c, (double)dt, (double)cfl, steps, verification_freq));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "  Mode                  : Data Assimilation%s\n"
                            "  Flux scheme           : %s\n"
                            "  Grid dimensions       : %" PetscInt_FMT " x %" PetscInt_FMT "\n"
                            "  State dimension       : %" PetscInt_FMT " (%" PetscInt_FMT " grid points x %d DOF)\n"
                            "  Observation dimension : %" PetscInt_FMT "\n"
                            "  Observation stride    : %" PetscInt_FMT "\n"
                            "  Ensemble size         : %" PetscInt_FMT "\n"
                            "  Domain size           : %.2f x %.2f\n"
                            "  Grid spacing          : dx=%.4f, dy=%.4f\n"
                            "  Mean height (h0)      : %.4f\n"
                            "  Wave amplitudes       : Ax=%.4f, Ay=%.4f\n"
                            "  Gravitational const   : %.4f\n"
                            "  Wave speed (c)        : %.4f\n"
                            "  Time step (dt)        : %.4f\n"
                            "  CFL number            : %.4f\n"
                            "  Total steps           : %" PetscInt_FMT "\n"
                            "  Observation frequency : %" PetscInt_FMT "\n"
                            "  Observation noise std : %.3f\n"
                            "  Random seed           : %" PetscInt_FMT "\n",
                            enable_verification ? " with verification" : "", flux_name, nx, ny, nx * ny * ndof, nx * ny, (int)ndof, nobs, obs_stride, ensemble_size, (double)Lx, (double)Ly, (double)dx, (double)dy, (double)h0, (double)Ax, (double)Ay, (double)g, (double)c, (double)dt, (double)cfl, steps, obs_freq, (double)obs_error_std, random_seed));
      if (enable_verification) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Verification freq     : %" PetscInt_FMT "\n", verification_freq));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    }
  }

  /* Open output file if requested - only in serial mode */
  if (output_enabled) {
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    if (size > 1) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Output file generation is only supported in serial mode (currently running with %d processes)\n", (int)size));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "         Disabling output file. Run with single process to enable.\n\n"));
      output_enabled = PETSC_FALSE;
      fp             = NULL;
    } else {
      PetscCall(PetscFOpen(PETSC_COMM_WORLD, output_file, "w", &fp));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# 2D Shallow Water LETKF Output\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# nx=%d, ny=%d, ndof=%d, nobs=%d, ensemble_size=%d\n", (int)nx, (int)ny, (int)ndof, (int)nobs, (int)ensemble_size));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# dt=%.6f, g=%.6f, obs_error_std=%.6f\n", (double)dt, (double)g, (double)obs_error_std));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# Format: step time [truth]x(nx*ny*ndof) [mean]x(nx*ny*ndof) [obs]x(nobs) rmse_forecast rmse_analysis\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Writing output to: %s\n\n", output_file));
    }
  }

  /* Print initial condition */
  if (verification_mode) {
    PetscReal L1_err, L2_err, Linf_err;
    PetscCall(ComputeAnalyticError(x0, da_state, 0.0, Lx, Ly, g, h0, Ax, Ay, &L1_err, &L2_err, &Linf_err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4d, time %6.3f  L1=%.5e  L2=%.5e  Linf=%.5e [initial]\n", 0, 0.0, (double)L1_err, (double)L2_err, (double)Linf_err));
  } else {
    PetscReal rmse_initial;
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
    PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, nx * ny * ndof, &rmse_initial));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4d, time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f [initial]\n", 0, 0.0, (double)rmse_initial, (double)rmse_initial));

    if (output_enabled && fp) {
      const PetscScalar *truth_array, *mean_array;
      PetscInt           i;
      PetscCall(VecGetArrayRead(truth_state, &truth_array));
      PetscCall(VecGetArrayRead(x_mean, &mean_array));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "0 0.000000"));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(truth_array[i])));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(mean_array[i])));
      for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " nan"));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e %.8e\n", (double)rmse_initial, (double)rmse_initial));
      PetscCall(VecRestoreArrayRead(truth_state, &truth_array));
      PetscCall(VecRestoreArrayRead(x_mean, &mean_array));
    }
  }

  /* Main simulation loop */
  if (verification_mode) {
    /* Pure verification mode: advance and compute errors */
    Vec x_numerical;
    PetscCall(VecDuplicate(x0, &x_numerical));
    PetscCall(VecCopy(x0, x_numerical));

    /* Write initial condition to file */
    if (output_enabled && fp) {
      const PetscScalar *num_array;
      PetscInt           i;
      PetscCall(VecGetArrayRead(x_numerical, &num_array));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "0 0.000000"));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(num_array[i])));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(num_array[i])));
      for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " nan"));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " 0.0 0.0\n"));
      PetscCall(VecRestoreArrayRead(x_numerical, &num_array));
    }

    for (step = 1; step <= steps; step++) {
      PetscReal time = step * dt;

      /* Advance numerical solution */
      PetscCall(ShallowWaterStep2D(x_numerical, x_numerical, sw_ctx));

      /* Compute error against analytic solution */
      if (step % verification_freq == 0 || step == steps) {
        PetscReal L1_err, L2_err, Linf_err;
        PetscCall(ComputeAnalyticError(x_numerical, da_state, time, Lx, Ly, g, h0, Ax, Ay, &L1_err, &L2_err, &Linf_err));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  L1=%.5e  L2=%.5e  Linf=%.5e\n", step, (double)time, (double)L1_err, (double)L2_err, (double)Linf_err));
      }

      /* Write data to output file */
      if (output_enabled && fp) {
        const PetscScalar *num_array;
        PetscInt           i;
        PetscCall(VecGetArrayRead(x_numerical, &num_array));
        PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "%d %.6f", (int)step, (double)time));
        /* Write numerical solution as both truth and mean */
        for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(num_array[i])));
        for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(num_array[i])));
        for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " nan"));
        PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " 0.0 0.0\n"));
        PetscCall(VecRestoreArrayRead(x_numerical, &num_array));
      }
    }
    PetscCall(VecDestroy(&x_numerical));
  } else {
    /* Data assimilation mode */
    for (step = 1; step <= steps; step++) {
      PetscReal time = step * dt;

      /* Propagate ensemble and truth trajectory */
      PetscCall(PetscDAEnsembleForecast(da, ShallowWaterStep2D, sw_ctx));
      PetscCall(ShallowWaterStep2D(truth_state, truth_state, sw_ctx));

      /* Forecast step: compute ensemble mean and forecast RMSE */
      PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
      PetscCall(VecCopy(x_mean, x_forecast));
      PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, nx * ny * ndof, &rmse_forecast));
      rmse_analysis = rmse_forecast;

      /* Analysis step: assimilate observations when available */
      if (step % obs_freq == 0 && step > 0) {
        Vec truth_obs, temp_truth;
        PetscCall(MatCreateVecs(H, NULL, &truth_obs));
        PetscCall(MatCreateVecs(H, &temp_truth, NULL));

        /* Generate observations from truth */
        PetscCall(VecCopy(truth_state, temp_truth));
        PetscCall(MatMult(H, temp_truth, truth_obs));

        /* Add observation noise */
        PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
        PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_obs));

        /* Perform LETKF analysis */
        PetscCall(PetscDAEnsembleAnalysis(da, observation, H));

        /* Clean up */
        PetscCall(VecDestroy(&temp_truth));
        PetscCall(VecDestroy(&truth_obs));

        /* Compute analysis RMSE */
        PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
        PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, nx * ny * ndof, &rmse_analysis));
        obs_count++;
      }

      /* Compute verification errors if enabled */
      if (enable_verification && (step % verification_freq == 0)) {
        PetscReal L1_err, L2_err, Linf_err;
        PetscCall(ComputeAnalyticError(x_mean, da_state, time, Lx, Ly, g, h0, Ax, Ay, &L1_err, &L2_err, &Linf_err));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "                        Verification: L1=%.5e  L2=%.5e  Linf=%.5e\n", (double)L1_err, (double)L2_err, (double)Linf_err));
      }

      /* Accumulate statistics */
      sum_rmse_forecast += rmse_forecast;
      sum_rmse_analysis += rmse_analysis;
      n_stat_steps++;

      /* Write data to output file */
      if (output_enabled && fp) {
        const PetscScalar *truth_array, *mean_array, *obs_array;
        PetscInt           i;
        PetscCall(VecGetArrayRead(truth_state, &truth_array));
        PetscCall(VecGetArrayRead(x_mean, &mean_array));
        PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "%d %.6f", (int)step, (double)time));
        for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(truth_array[i])));
        for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(mean_array[i])));
        if (step % obs_freq == 0 && step > 0) {
          PetscCall(VecGetArrayRead(observation, &obs_array));
          for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(obs_array[i])));
          PetscCall(VecRestoreArrayRead(observation, &obs_array));
        } else {
          for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " nan"));
        }
        PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e %.8e\n", (double)rmse_forecast, (double)rmse_analysis));
        PetscCall(VecRestoreArrayRead(truth_state, &truth_array));
        PetscCall(VecRestoreArrayRead(x_mean, &mean_array));
      }

      /* Progress reporting */
      if (progress_freq == 0) {
        if (step == steps) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis));
      } else {
        if ((step % progress_freq == 0) || (step == steps)) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis));
      }
    }
  }

  /* Report final statistics */
  if (verification_mode) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nVerification test complete.\n"));
  } else {
    if (n_stat_steps > 0) {
      PetscReal avg_rmse_forecast = sum_rmse_forecast / n_stat_steps;
      PetscReal avg_rmse_analysis = sum_rmse_analysis / n_stat_steps;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStatistics (%" PetscInt_FMT " steps):\n", n_stat_steps));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==================================================\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (forecast) : %.5f\n", (double)avg_rmse_forecast));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (analysis) : %.5f\n", (double)avg_rmse_analysis));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Observations used    : %" PetscInt_FMT "\n\n", obs_count));
    }
  }

  /* Close output file */
  if (output_enabled && fp) {
    PetscCall(PetscFClose(PETSC_COMM_WORLD, fp));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Output written to: %s\n", output_file));
  }

  /* Cleanup */
  if (!verification_mode) {
    PetscCall(MatDestroy(&H));
    PetscCall(MatDestroy(&H1));
    PetscCall(VecDestroy(&x_forecast));
    PetscCall(VecDestroy(&x_mean));
    PetscCall(VecDestroy(&obs_error_var));
    PetscCall(VecDestroy(&obs_noise));
    PetscCall(VecDestroy(&observation));
    PetscCall(PetscDADestroy(&da));
  }
  /* These are created in both modes */
  PetscCall(VecDestroy(&rmse_work));
  PetscCall(VecDestroy(&truth_state));
  PetscCall(VecDestroy(&x0));
  PetscCall(DMDestroy(&da_state));
  PetscCall(ShallowWater2DContextDestroy(&sw_ctx));
  PetscCall(PetscRandomDestroy(&rng));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: kokkos_kernels !complex
    args: -steps 10 -progress_freq 1 -petscda_view -petscda_ensemble_size 10 -obs_freq 2 -obs_error 0.03 -nx 21 -ny 21

    test:
      suffix: letkf_wave2d
      args: -petscda_type letkf -petscda_ensemble_size 7

    test:
      nsize: 3
      suffix: kokkos_wave2d
      args: -petscda_type letkf -mat_type aijkokkos -vec_type kokkos -petscda_ensemble_size 5 -petscda_letkf_obs_per_vertex 5

  test:
    suffix: verification
    requires: !complex kokkos_kernels
    args: -verification_mode -steps 50 -nx 40 -ny 40 -verification_freq 10

  test:
    suffix: verification_parallel
    requires: !complex kokkos_kernels
    nsize: 2
    args: -verification_mode -steps 20 -nx 30 -ny 30 -verification_freq 5

TEST*/
