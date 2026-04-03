static char help[] = "2D shallow water LETKF data assimilation example.\n"
                     "Implements 2D shallow water equations with 3 DOF per grid point (h, hu, hv).\n\n"
                     "Usage:\n"
                     "  ./ex4 -steps 100 -nx 41 -ny 41 -petscda_type letkf -ensemble_size 30\n\n";

#include <petscda.h>
#include <petscdmda.h>
#include <petscts.h>

#include "ex4common.h"

/* Default parameter values */
#define DEFAULT_NX                      40
#define DEFAULT_NY                      40
#define DEFAULT_STEPS                   100
#define DEFAULT_OBS_FREQ                5
#define DEFAULT_RANDOM_SEED             12345
#define DEFAULT_G                       9.81
#define DEFAULT_DT                      0.02
#define DEFAULT_LX                      80.0
#define DEFAULT_LY                      80.0
#define DEFAULT_H0                      1.5
#define DEFAULT_AX                      0.2
#define DEFAULT_AY                      0.2
#define DEFAULT_OBS_ERROR_STD           0.01
#define DEFAULT_INIT_PERTURB_STD        0.05
#define DEFAULT_INIT_H_BIAS             0.0
#define DEFAULT_ENSEMBLE_SIZE           30
#define DEFAULT_PROGRESS_FREQ           10
#define DEFAULT_OBS_STRIDE              2
#define DEFAULT_FALLBACK_OBS_PER_VERTEX 49
#define SPINUP_STEPS                    0

/* Minimum valid parameter values */
#define MIN_ENSEMBLE_SIZE 2
#define MIN_OBS_FREQ      1

/*
  CreateObservationMatrix2D - Create observation matrix H for 2D shallow water

  Observes water height (h) at every obs_stride-th grid point in both x and y directions.
*/
static PetscErrorCode CreateObservationMatrix2D(PetscInt nx, PetscInt ny, PetscInt ndof, PetscInt obs_stride, Mat *H, Mat *H1, PetscInt *nobs_out)
{
  PetscInt i, j, obs_idx;
  PetscInt nobs_x, nobs_y, nobs;
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  /* Calculate number of observations */
  nobs_x = (nx + obs_stride - 1) / obs_stride;
  nobs_y = (ny + obs_stride - 1) / obs_stride;
  nobs   = nobs_x * nobs_y;

  /* Create observation matrix H (nobs x nx*ny*ndof) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nobs, nx * ny * ndof, 1, NULL, 1, NULL, H));
  PetscCall(MatSetFromOptions(*H));

  /* Create H1 for scalar field (nobs x nx*ny) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nobs, nx * ny, 1, NULL, 1, NULL, H1));
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
  PetscInt i, j, rstart, rend;

  PetscFunctionBeginUser;
  /* Create Q matrix (nx*ny x nobs) - global/no localization for simplicity */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nx * ny, nobs, nobs, NULL, 0, NULL, Q));
  PetscCall(MatSetFromOptions(*Q));
  PetscCall(MatGetOwnershipRange(*Q, &rstart, &rend));

  /* Initialize with no localization: each state variable uses all observations */
  for (i = rstart; i < rend; i++) {
    for (j = 0; j < nobs; j++) PetscCall(MatSetValue(*Q, i, j, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !defined(PETSC_HAVE_KOKKOS_KERNELS)
/*
  CreateNearestObservationLocalization2D - Build a fixed-width localization pattern.

  Each state row connects to exactly n_obs_vertex nearby observations so the LETKF
  localization contract is satisfied even without the distance-based helper.
*/
static PetscReal GaspariCohnWeight(PetscReal r)
{
  PetscFunctionBeginHot;
  if (r >= 2.0) PetscFunctionReturn(0.0);
  if (r <= 1.0) PetscFunctionReturn((((-0.25 * r + 0.5) * r + 0.625) * r - 5.0 / 3.0) * r * r + 1.0);
  PetscFunctionReturn(((((r / 12.0 - 0.5) * r + 0.625) * r + 5.0 / 3.0) * r - 5.0) * r + 4.0 - 2.0 / (3.0 * r));
}

static PetscErrorCode CreateNearestObservationLocalization2D(PetscInt nx, PetscInt ny, PetscInt obs_stride, PetscInt nobs_vertex, Mat *Q)
{
  PetscInt rstart, rend;
  PetscInt nobs_x, nobs_y, nobs;

  PetscFunctionBeginUser;
  nobs_x = (nx + obs_stride - 1) / obs_stride;
  nobs_y = (ny + obs_stride - 1) / obs_stride;
  nobs   = nobs_x * nobs_y;

  PetscCheck(nobs_vertex > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of observations per vertex must be positive");
  PetscCheck(nobs_vertex <= nobs, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observations per vertex (%" PetscInt_FMT ") cannot exceed total observations (%" PetscInt_FMT ")", nobs_vertex, nobs);

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nx * ny, nobs, nobs_vertex, NULL, 0, NULL, Q));
  PetscCall(MatSetFromOptions(*Q));
  PetscCall(MatGetOwnershipRange(*Q, &rstart, &rend));

  for (PetscInt row = rstart; row < rend; row++) {
    PetscInt  gi       = row % nx;
    PetscInt  gj       = row / nx;
    PetscInt  selected = 0;
    PetscReal best_dist[64];
    PetscInt  best_col[64];

    PetscCheck(nobs_vertex <= 64, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Fallback nearest-observation localization currently supports up to 64 observations per vertex");
    for (PetscInt k = 0; k < nobs_vertex; k++) {
      best_dist[k] = PETSC_MAX_REAL;
      best_col[k]  = -1;
    }

    for (PetscInt oy = 0; oy < nobs_y; oy++) {
      PetscInt oj = PetscMin(oy * obs_stride, ny - 1);

      for (PetscInt ox = 0; ox < nobs_x; ox++) {
        PetscInt  oi  = PetscMin(ox * obs_stride, nx - 1);
        PetscInt  col = oy * nobs_x + ox;
        PetscInt  dxp = PetscAbsInt(gi - oi);
        PetscInt  dyp = PetscAbsInt(gj - oj);
        PetscReal dist;

        /* Respect the periodic geometry used by the DMDA and PDE model. */
        dxp  = PetscMin(dxp, nx - dxp);
        dyp  = PetscMin(dyp, ny - dyp);
        dist = (PetscReal)(dxp * dxp + dyp * dyp);

        if (selected < nobs_vertex) {
          best_dist[selected] = dist;
          best_col[selected]  = col;
          selected++;
        } else {
          PetscInt worst = 0;
          for (PetscInt k = 1; k < nobs_vertex; k++) {
            if (best_dist[k] > best_dist[worst]) worst = k;
          }
          if (dist < best_dist[worst]) {
            best_dist[worst] = dist;
            best_col[worst]  = col;
          }
        }
      }
    }

    {
      PetscReal radius = PetscSqrtReal(best_dist[0]);

      for (PetscInt k = 1; k < nobs_vertex; k++) radius = PetscMax(radius, PetscSqrtReal(best_dist[k]));
      radius = PetscMax(radius, 1.0);

      for (PetscInt k = 0; k < nobs_vertex; k++) {
        PetscReal dist   = PetscSqrtReal(best_dist[k]);
        PetscReal weight = GaspariCohnWeight(2.0 * dist / radius);

        PetscCall(MatSetValue(*Q, row, best_col[k], weight, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

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

static PetscErrorCode InitializeBalancedEnsemble(PetscDA da, DM da_state, PetscRandom rng, PetscInt nx, PetscInt ny, PetscInt ensemble_size, PetscReal Lx, PetscReal Ly, PetscReal g, PetscReal h0, PetscReal Ax, PetscReal Ay, PetscReal init_perturb_std, PetscReal init_h_bias)
{
  Vec        member;
  PetscReal *alpha_x = NULL, *alpha_y = NULL, *beta_x = NULL, *beta_y = NULL;
  PetscReal  mean_ax = 0.0, mean_ay = 0.0, mean_bx = 0.0, mean_by = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc4(ensemble_size, &alpha_x, ensemble_size, &alpha_y, ensemble_size, &beta_x, ensemble_size, &beta_y));

  for (PetscInt e = 0; e < ensemble_size; e++) {
    PetscReal r;

    PetscCall(PetscRandomGetValueReal(rng, &r));
    alpha_x[e] = init_perturb_std * r;
    mean_ax += alpha_x[e];
    PetscCall(PetscRandomGetValueReal(rng, &r));
    alpha_y[e] = init_perturb_std * r;
    mean_ay += alpha_y[e];
    PetscCall(PetscRandomGetValueReal(rng, &r));
    beta_x[e] = init_perturb_std * r;
    mean_bx += beta_x[e];
    PetscCall(PetscRandomGetValueReal(rng, &r));
    beta_y[e] = init_perturb_std * r;
    mean_by += beta_y[e];
  }

  mean_ax /= ensemble_size;
  mean_ay /= ensemble_size;
  mean_bx /= ensemble_size;
  mean_by /= ensemble_size;

  PetscCall(DMCreateGlobalVector(da_state, &member));
  for (PetscInt e = 0; e < ensemble_size; e++) {
    PetscScalar ***x_array;
    PetscInt       xs, ys, xm, ym;
    PetscReal      axp = alpha_x[e] - mean_ax;
    PetscReal      ayp = alpha_y[e] - mean_ay;
    PetscReal      bxp = beta_x[e] - mean_bx;
    PetscReal      byp = beta_y[e] - mean_by;
    PetscReal      dx  = Lx / nx;
    PetscReal      dy  = Ly / ny;
    PetscReal      kx  = 2.0 * PETSC_PI / Lx;
    PetscReal      ky  = 2.0 * PETSC_PI / Ly;
    PetscReal      c   = PetscSqrtReal(g * h0);

    PetscCall(DMDAGetCorners(da_state, &xs, &ys, NULL, &xm, &ym, NULL));
    PetscCall(DMDAVecGetArrayDOF(da_state, member, &x_array));
    for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {
        PetscReal x  = ((PetscReal)i + 0.5) * dx;
        PetscReal y  = ((PetscReal)j + 0.5) * dy;
        PetscReal sx = PetscSinReal(kx * x), cx = PetscCosReal(kx * x);
        PetscReal sy = PetscSinReal(ky * y), cy = PetscCosReal(ky * y);
        PetscReal eta_x = axp * sx + bxp * cx;
        PetscReal eta_y = ayp * sy + byp * cy;
        PetscReal eta   = eta_x + eta_y;
        PetscReal u     = (c / h0) * (axp * cx - bxp * sx);
        PetscReal v     = (c / h0) * (ayp * cy - byp * sy);
        PetscReal hbase, hubase, hvbase;

        PetscCall(ShallowWaterSolution_Wave2D(Lx, Ly, x, y, 0.0, g, h0, Ax, Ay, &hbase, &hubase, &hvbase));
        x_array[j][i][0] = hbase + init_h_bias + eta;
        x_array[j][i][1] = hubase + h0 * u;
        x_array[j][i][2] = hvbase + h0 * v;
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(da_state, member, &x_array));
    PetscCall(PetscDAEnsembleSetMember(da, e, member));
  }

  PetscCall(VecDestroy(&member));
  PetscCall(PetscFree4(alpha_x, alpha_y, beta_x, beta_y));
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
  PetscReal      init_perturb_std        = DEFAULT_INIT_PERTURB_STD;
  PetscReal      init_h_bias             = DEFAULT_INIT_H_BIAS;
  PetscReal      obs_error_std           = DEFAULT_OBS_ERROR_STD;
  PetscBool      use_fake_localization   = PETSC_FALSE;
  PetscBool      obs_per_vertex_set      = PETSC_FALSE;
  PetscInt       num_observations_vertex = 7;
  Ex4FluxType    flux_type               = EX4_FLUX_RUSANOV;
  char           output_file[PETSC_MAX_PATH_LEN];
  PetscBool      output_enabled = PETSC_FALSE;
  FILE          *fp             = NULL;

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
  PetscReal rmse_initial  = 0.0;
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
  PetscCall(PetscOptionsReal("-init_perturb_std", "Initial ensemble perturbation standard deviation", "", init_perturb_std, &init_perturb_std, NULL));
  PetscCall(PetscOptionsReal("-init_h_bias", "Initial ensemble-mean bias applied to height", "", init_h_bias, &init_h_bias, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation error standard deviation", "", obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble perturbations", "", random_seed, &random_seed, NULL));
  PetscCall(PetscOptionsInt("-progress_freq", "Print progress every N steps (0 = only first/last)", "", progress_freq, &progress_freq, NULL));
  PetscCall(PetscOptionsString("-output_file", "Output file for visualization data", "", "", output_file, sizeof(output_file), &output_enabled));
  PetscCall(PetscOptionsEnum("-ex4_flux", "Flux scheme (rusanov/mc)", "", Ex4FluxTypes, (PetscEnum)flux_type, (PetscEnum *)&flux_type, NULL));
  PetscCall(PetscOptionsBool("-use_fake_localization", "Use fake localization matrix", "", use_fake_localization, &use_fake_localization, NULL));
  PetscCall(PetscOptionsInt("-petscda_letkf_obs_per_vertex", "Number of observations per vertex", "", num_observations_vertex, &num_observations_vertex, &obs_per_vertex_set));
  PetscOptionsEnd();

  /* Calculate number of observations */
  nobs = ((nx + obs_stride - 1) / obs_stride) * ((ny + obs_stride - 1) / obs_stride);

  /* Set num_observations_vertex for fake localization after nobs is calculated */
  if (use_fake_localization && !obs_per_vertex_set) num_observations_vertex = nobs;

  PetscCall(ValidateParameters(&nx, &ny, &nobs, &steps, &obs_freq, &ensemble_size, &dt, &g, &obs_error_std));
  PetscCheck(init_perturb_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Initial perturbation std must be positive");

  /* Create 2D periodic DMDA with 3 DOF (h, hu, hv) */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, nx, ny, PETSC_DECIDE, PETSC_DECIDE, ndof, 2, NULL, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  /* Create shallow water context */
  PetscCall(ShallowWater2DContextCreate(da_state, nx, ny, Lx, Ly, g, dt, h0, Ax, Ay, PETSC_FALSE, flux_type, &sw_ctx));

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

  if (init_h_bias != 0.0) {
    PetscScalar ***x_array;
    PetscInt       xs, ys, xm, ym, i, j;

    PetscCall(DMDAGetCorners(da_state, &xs, &ys, NULL, &xm, &ym, NULL));
    PetscCall(DMDAVecGetArrayDOF(da_state, x0, &x_array));
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) x_array[j][i][0] += init_h_bias;
    }
    PetscCall(DMDAVecRestoreArrayDOF(da_state, x0, &x_array));
  }

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

  /* Create observation matrix H */
  PetscCall(CreateObservationMatrix2D(nx, ny, ndof, obs_stride, &H, &H1, &nobs));

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

  /* LETKF requires the symmetric eigen square root for the local update. */
  {
    PetscBool isletkf;
    PetscCall(PetscObjectTypeCompare((PetscObject)da, PETSCDALETKF, &isletkf));
    if (isletkf) PetscCall(PetscDAEnsembleSetSqrtType(da, PETSCDA_SQRT_EIGEN));
  }

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
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
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

      PetscCheck(cdof >= 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Coordinate DM block size must be at least 2 for 2D localization");

      /* Extract x/y coordinate fields into scalar vectors expected by localization setup. */
      for (PetscInt d = 0; d < 2; d++) {
        PetscCall(VecCreate(PETSC_COMM_WORLD, &Vecxyz[d]));
        PetscCall(VecSetSizes(Vecxyz[d], PETSC_DECIDE, nx * ny));
        PetscCall(VecSetFromOptions(Vecxyz[d]));
        PetscCall(PetscObjectSetName((PetscObject)Vecxyz[d], d == 0 ? "x_coordinate" : "y_coordinate"));
        PetscCall(VecStrideGather(coord, d, Vecxyz[d], INSERT_VALUES));
      }

      /* Get localization matrix using distance-based method */
      PetscCall(PetscDALETKFGetLocalizationMatrix(num_observations_vertex, 1, Vecxyz, bd, H1, &Q));
      PetscCall(VecDestroy(&Vecxyz[0]));
      PetscCall(VecDestroy(&Vecxyz[1]));
      PetscCall(PetscDALETKFSetObsPerVertex(da, num_observations_vertex));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using distance-based localization with %" PetscInt_FMT " observations per vertex\n", num_observations_vertex));
#else
      PetscInt fallback_obs_per_vertex = num_observations_vertex;

      if (!obs_per_vertex_set) fallback_obs_per_vertex = PetscMax(fallback_obs_per_vertex, DEFAULT_FALLBACK_OBS_PER_VERTEX);
      fallback_obs_per_vertex = PetscMax(fallback_obs_per_vertex, ensemble_size);
      fallback_obs_per_vertex = PetscMin(fallback_obs_per_vertex, nobs);

      PetscCall(CreateNearestObservationLocalization2D(nx, ny, obs_stride, fallback_obs_per_vertex, &Q));
      PetscCall(PetscDALETKFSetObsPerVertex(da, fallback_obs_per_vertex));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Distance-based localization unavailable in this build; using nearest-observation localization with %" PetscInt_FMT " observations per vertex instead\n", fallback_obs_per_vertex));
#endif
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
  PetscCall(InitializeBalancedEnsemble(da, da_state, rng, nx, ny, ensemble_size, Lx, Ly, g, h0, Ax, Ay, init_perturb_std, init_h_bias));
  PetscCall(PetscDAViewFromOptions(da, NULL, "-petscda_view"));

  /* Print configuration summary */
  {
    const char *flux_name = (flux_type == EX4_FLUX_RUSANOV) ? "Rusanov (1st order)" : "MC (2nd order)";
    const char *mode_name = "LETKF";
    PetscReal   dx        = Lx / nx;
    PetscReal   dy        = Ly / ny;
    PetscReal   c         = PetscSqrtReal(g * h0);
    PetscReal   cfl       = dt * c * (1.0 / dx + 1.0 / dy);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D Shallow Water %s Example\n", mode_name));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==============================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "  Mode                  : Data Assimilation\n"
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
                          "  Init perturb std      : %.3f\n"
                          "  Init height bias      : %.3f\n"
                          "  Observation noise std : %.3f\n"
                          "  Random seed           : %" PetscInt_FMT "\n",
                          flux_name, nx, ny, nx * ny * ndof, nx * ny, (int)ndof, nobs, obs_stride, ensemble_size, (double)Lx, (double)Ly, (double)dx, (double)dy, (double)h0, (double)Ax, (double)Ay, (double)g, (double)c, (double)dt, (double)cfl, steps, obs_freq, (double)init_perturb_std, (double)init_h_bias, (double)obs_error_std, random_seed));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
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
  PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
  PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, nx * ny * ndof, &rmse_initial));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f [initial]\n", (PetscInt)0, 0.0, (double)rmse_initial, (double)rmse_initial));

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

  /* Main simulation loop */
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
      } else
        for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " nan"));
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

  /* Report final statistics */
  if (n_stat_steps > 0) {
    PetscReal avg_rmse_forecast = sum_rmse_forecast / n_stat_steps;
    PetscReal avg_rmse_analysis = sum_rmse_analysis / n_stat_steps;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStatistics (%" PetscInt_FMT " steps):\n", n_stat_steps));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==================================================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (forecast) : %.5f\n", (double)avg_rmse_forecast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (analysis) : %.5f\n", (double)avg_rmse_analysis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Observations used    : %" PetscInt_FMT "\n\n", obs_count));
  }

  /* Close output file */
  if (output_enabled && fp) {
    PetscCall(PetscFClose(PETSC_COMM_WORLD, fp));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Output written to: %s\n", output_file));
  }

  /* Cleanup */
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&H1));
  PetscCall(VecDestroy(&x_forecast));
  PetscCall(VecDestroy(&x_mean));
  PetscCall(VecDestroy(&obs_error_var));
  PetscCall(VecDestroy(&obs_noise));
  PetscCall(VecDestroy(&observation));
  PetscCall(PetscDADestroy(&da));
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
TEST*/
