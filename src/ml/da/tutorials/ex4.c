static char help[] = "2D shallow water LETKF data assimilation example.\n"
                     "Implements 2D shallow water equations with 3 DOF per grid point (h, hu, hv).\n\n"
                     "Usage:\n"
                     "  ./ex4 -steps 100 -nx 41 -ny 41 -petscda_type letkf -petscda_ensemble_size 30\n\n";

#include <petscda.h>
#include <petscdmda.h>
#include <petscts.h>

#include "ex4.h"

/* Default parameter values */
#define DEFAULT_NX                     40
#define DEFAULT_NY                     40
#define DEFAULT_STEPS                  100
#define DEFAULT_OBS_FREQ               5
#define DEFAULT_RANDOM_SEED            12345
#define DEFAULT_G                      9.81
#define DEFAULT_DT                     0.02
#define DEFAULT_LX                     80.0
#define DEFAULT_LY                     80.0
#define DEFAULT_H0                     1.5
#define DEFAULT_AX                     0.2
#define DEFAULT_AY                     0.2
#define DEFAULT_OBS_ERROR_STD          0.01
#define DEFAULT_INIT_PERTURB_AMPLITUDE 0.05
#define DEFAULT_INIT_H_BIAS            0.0
#define DEFAULT_ENSEMBLE_SIZE          30
#define DEFAULT_PROGRESS_FREQ          10
#define DEFAULT_OBS_STRIDE             2
#define DEFAULT_LOCALIZATION_RADIUS    20.0 /* kernel half-width; effective cutoff is 2*radius for gaspari_cohn/gaussian and radius for boxcar (~10 obs spacings on the default 80x80 domain with obs_stride=2) */
#define SPINUP_STEPS                   0

/* Minimum valid parameter values */
#define MIN_ENSEMBLE_SIZE 2
#define MIN_OBS_FREQ      1

/*
  ShallowWaterStep2D - PetscDAEnsembleForecast callback that advances every column of a dense ensemble Mat by one TS step.
*/
static PetscErrorCode ShallowWaterStep2D(Mat ensemble, PetscCtx ctx)
{
  ShallowWater2DCtx *sw = (ShallowWater2DCtx *)ctx;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(ensemble, NULL, &n));
  /* Collective: dense ensemble Mat is row-distributed, so every rank visits every global column j and
     MatDenseGetColumnVec returns the parallel column-Vec that all ranks step together. Use the
     read-write variant because TSSolve reads the column as the initial condition and writes the
     stepped solution back; the write-only variant would skip the device->host sync on device-backed
     dense (MATSEQDENSECUDA/HIP, MATMPIDENSECUDA/HIP) and feed TSSolve stale host data.
     MMS forcing is not exercised here: main() leaves cfg.verify_mms = PETSC_FALSE, so t_start = 0.0
     is correct for this autonomous RHS. */
  for (PetscInt j = 0; j < n; j++) {
    Vec col;

    PetscCall(MatDenseGetColumnVec(ensemble, j, &col));
    PetscCall(ShallowWaterStep2DVec(sw, 0.0, col));
    PetscCall(MatDenseRestoreColumnVec(ensemble, j, &col));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateObservationMatrix2D - Create observation matrix H for 2D shallow water

  Observes water height (h) at every obs_stride-th grid point in both x and y directions.
*/
static PetscErrorCode CreateObservationMatrix2D(PetscInt nx, PetscInt ny, PetscInt ndof, PetscInt obs_stride, PetscInt local_state_size, Mat *H, Mat *H1, PetscInt *nobs_out)
{
  PetscInt i, j, obs_idx;
  PetscInt nobs_x, nobs_y, nobs;
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  /* Calculate number of observations */
  nobs_x = (nx + obs_stride - 1) / obs_stride;
  nobs_y = (ny + obs_stride - 1) / obs_stride;
  nobs   = nobs_x * nobs_y;

  /* Create observation matrix H (nobs x nx*ny*ndof). The column local size must match
     the DMDA-partitioned state vector so that MatMult(H, state, obs) is well-defined. */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, local_state_size, nobs, nx * ny * ndof, 1, NULL, 1, NULL, H));
  PetscCall(MatSetFromOptions(*H));

  /* Create H1 for scalar field (nobs x nx*ny); the column local size must match the
     per-grid-point coordinate vectors used for localization. */
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

static PetscErrorCode InitializeBalancedEnsemble(PetscDA da, DM da_state, ShallowWater2DCtx *sw_ctx, PetscInt random_seed, PetscInt ensemble_size, PetscReal init_perturb_amplitude, PetscReal init_h_bias)
{
  Vec         member;
  PetscRandom coef_rng;
  PetscInt    nx = sw_ctx->nx, ny = sw_ctx->ny;
  PetscReal  *alpha_x, *alpha_y, *beta_x, *beta_y;
  PetscReal   mean_ax = 0.0, mean_ay = 0.0, mean_bx = 0.0, mean_by = 0.0;
  PetscReal   Lx = sw_ctx->Lx, Ly = sw_ctx->Ly, g = sw_ctx->g, h0 = sw_ctx->h0, Ax = sw_ctx->Ax, Ay = sw_ctx->Ay;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc4(ensemble_size, &alpha_x, ensemble_size, &alpha_y, ensemble_size, &beta_x, ensemble_size, &beta_y));

  /* The per-member coefficients are sampled redundantly on every rank because they parameterize a
     globally smooth perturbation written into a parallel Vec. Use a PETSC_COMM_SELF rng with a
     rank-independent seed so that all ranks observe identical coefficient sequences and the
     ensemble member fields remain continuous across rank partitions. */
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &coef_rng));
  PetscCall(PetscRandomSetSeed(coef_rng, (unsigned long)random_seed));
  PetscCall(PetscRandomSeed(coef_rng));

  for (PetscInt e = 0; e < ensemble_size; e++) {
    PetscReal r;

    PetscCall(PetscRandomGetValueReal(coef_rng, &r));
    alpha_x[e] = init_perturb_amplitude * r;
    mean_ax += alpha_x[e];
    PetscCall(PetscRandomGetValueReal(coef_rng, &r));
    alpha_y[e] = init_perturb_amplitude * r;
    mean_ay += alpha_y[e];
    PetscCall(PetscRandomGetValueReal(coef_rng, &r));
    beta_x[e] = init_perturb_amplitude * r;
    mean_bx += beta_x[e];
    PetscCall(PetscRandomGetValueReal(coef_rng, &r));
    beta_y[e] = init_perturb_amplitude * r;
    mean_by += beta_y[e];
  }
  PetscCall(PetscRandomDestroy(&coef_rng));

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
    PetscCall(DMDAVecGetArrayDOFWrite(da_state, member, &x_array));
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

        ShallowWaterSolution_Wave2D(Lx, Ly, x, y, 0.0, g, h0, Ax, Ay, &hbase, &hubase, &hvbase);
        x_array[j][i][0] = hbase + init_h_bias + eta;
        x_array[j][i][1] = hubase + h0 * u;
        x_array[j][i][2] = hvbase + h0 * v;
      }
    }
    PetscCall(DMDAVecRestoreArrayDOFWrite(da_state, member, &x_array));
    PetscCall(PetscDAEnsembleSetMember(da, e, member));
  }

  PetscCall(VecDestroy(&member));
  PetscCall(PetscFree4(alpha_x, alpha_y, beta_x, beta_y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ValidateParameters - Validate input parameters
*/
static PetscErrorCode ValidateParameters(PetscInt *nx, PetscInt *ny, PetscInt *steps, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *g, PetscReal *obs_error_std)
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

/* Wire LETKF localization (kernel type already chosen by SetFromOptions): build per-grid-point
   coordinate vectors, register them, honor a user-supplied radius via the options database, and
   print a summary. The NONE kernel needs no setup and is handled by the outer guard. */
static PetscErrorCode ConfigureLETKFLocalization(PetscDA da, DM da_state, PetscInt nx, PetscInt ny, PetscInt ndof, PetscInt local_state_size, PetscReal Lx, PetscReal Ly, Mat H1, PetscReal *localization_radius)
{
  PetscDALETKFLocalizationType loc_type;
  Vec                          xyz[3] = {NULL, NULL, NULL};
  Vec                          coord;
  DM                           cda;
  PetscReal                    bd[3] = {Lx, Ly, 0.0};
  PetscBool                    radius_set;
  const char                  *da_prefix, *kname = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscDALETKFGetLocalizationType(da, &loc_type));
  if (loc_type == PETSCDA_LETKF_LOC_NONE) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization disabled (LETKF NONE; equivalent to global ETKF)\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(DMDASetUniformCoordinates(da_state, 0.0, Lx, 0.0, Ly, 0.0, 0.0));
  PetscCall(DMGetCoordinateDM(da_state, &cda));
  PetscCall(DMGetCoordinates(da_state, &coord));

  /* xyz must share the DMDA per-grid-point partition so that VecStrideGather from
     the (block-2) coordinate vector lands in matching local rows. */
  for (PetscInt d = 0; d < 2; d++) {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &xyz[d]));
    PetscCall(VecSetSizes(xyz[d], local_state_size / ndof, nx * ny));
    PetscCall(VecSetFromOptions(xyz[d]));
    PetscCall(PetscObjectSetName((PetscObject)xyz[d], d == 0 ? "x_coordinate" : "y_coordinate"));
    PetscCall(VecStrideGather(coord, d, xyz[d], INSERT_VALUES));
  }

  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)da, &da_prefix));
  PetscCall(PetscOptionsHasName(NULL, da_prefix, "-petscda_letkf_localization_radius", &radius_set));
  if (!radius_set) PetscCall(PetscDALETKFSetLocalizationRadius(da, *localization_radius));
  PetscCall(PetscDALETKFGetLocalizationRadius(da, localization_radius));
  PetscCall(PetscDALETKFSetLocalizationCoordinates(da, xyz, bd, H1));
  for (PetscInt d = 0; d < 2; d++) PetscCall(VecDestroy(&xyz[d]));

  switch (loc_type) {
  case PETSCDA_LETKF_LOC_GASPARI_COHN:
    kname = "Gaspari-Cohn";
    break;
  case PETSCDA_LETKF_LOC_GAUSSIAN:
    kname = "Gaussian";
    break;
  case PETSCDA_LETKF_LOC_BOXCAR:
    kname = "boxcar";
    break;
  case PETSCDA_LETKF_LOC_NONE:
  case PETSCDA_LETKF_LOC_NUM_TYPES:
    break;
  }
  PetscCheck(kname, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unexpected localization type %d", (int)loc_type);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using %s localization with radius %g\n", kname, (double)*localization_radius));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  ShallowWater2DCtx   *sw_ctx = NULL;
  ShallowWater2DConfig cfg;
  DM                   da_state;
  PetscDA              da;
  Vec                  x0, x_mean, x_forecast;
  Vec                  truth_state, rmse_work;
  Vec                  observation, obs_noise, obs_error_var;
  Mat                  H, H1;
  PetscRandom          rng;
  FILE                *fp = NULL;
  char                 output_file[PETSC_MAX_PATH_LEN];
  const PetscInt       ndof = 3; /* h, hu, hv */
  PetscInt             nx = DEFAULT_NX, ny = DEFAULT_NY;
  PetscInt             steps = DEFAULT_STEPS, obs_freq = DEFAULT_OBS_FREQ, obs_stride = DEFAULT_OBS_STRIDE;
  PetscInt             ensemble_size = DEFAULT_ENSEMBLE_SIZE, n_spin = SPINUP_STEPS, random_seed = DEFAULT_RANDOM_SEED;
  PetscInt             progress_freq = DEFAULT_PROGRESS_FREQ;
  PetscInt             nobs = 0, n_stat_steps = 0, obs_count = 0, step;
  PetscInt             local_state_size, local_obs_size;
  PetscReal            g = DEFAULT_G, dt = DEFAULT_DT;
  PetscReal            Lx = DEFAULT_LX, Ly = DEFAULT_LY, h0 = DEFAULT_H0;
  PetscReal            Ax = DEFAULT_AX, Ay = DEFAULT_AY;
  PetscReal            init_perturb_amplitude = DEFAULT_INIT_PERTURB_AMPLITUDE, init_h_bias = DEFAULT_INIT_H_BIAS;
  PetscReal            obs_error_std       = DEFAULT_OBS_ERROR_STD;
  PetscReal            localization_radius = DEFAULT_LOCALIZATION_RADIUS;
  PetscReal            rmse_initial = 0.0, rmse_forecast = 0.0, rmse_analysis = 0.0, truth_time = 0.0;
  PetscReal            sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscReal            dx, dy, c, cfl;
  PetscBool            output_enabled = PETSC_FALSE;
  PetscMPIInt          rank;

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
  PetscCall(PetscOptionsReal("-init_perturb_amplitude", "Initial ensemble perturbation amplitude (uniform on [0, amplitude), then mean-centered)", "", init_perturb_amplitude, &init_perturb_amplitude, NULL));
  PetscCall(PetscOptionsReal("-init_h_bias", "Initial ensemble-mean bias applied to height", "", init_h_bias, &init_h_bias, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation error standard deviation", "", obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble perturbations", "", random_seed, &random_seed, NULL));
  PetscCall(PetscOptionsInt("-n_spin", "Number of spinup steps for the truth trajectory before assimilation starts", "", n_spin, &n_spin, NULL));
  PetscCall(PetscOptionsInt("-progress_freq", "Print progress every N steps (0 = only first/last)", "", progress_freq, &progress_freq, NULL));
  PetscCall(PetscOptionsString("-output_file", "Output file for visualization data", "", "", output_file, sizeof(output_file), &output_enabled));
  PetscOptionsEnd();

  PetscCall(ValidateParameters(&nx, &ny, &steps, &obs_freq, &ensemble_size, &dt, &g, &obs_error_std));
  PetscCheck(init_perturb_amplitude > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Initial perturbation amplitude must be positive");

  cfg.nx         = nx;
  cfg.ny         = ny;
  cfg.Lx         = Lx;
  cfg.Ly         = Ly;
  cfg.g          = g;
  cfg.dt         = dt;
  cfg.h0         = h0;
  cfg.Ax         = Ax;
  cfg.Ay         = Ay;
  cfg.verify_mms = PETSC_FALSE;
  PetscCall(SetupForwardProblem(&cfg, &da_state, &sw_ctx, &x0));

  /* Initialize random number generator */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  PetscCall(PetscRandomSetSeed(rng, (unsigned long)(random_seed + rank)));
  PetscCall(PetscRandomSetFromOptions(rng));
  PetscCall(PetscRandomSeed(rng));

  /* Set initial condition from analytic wave solution */
  PetscCall(SetInitialCondition(da_state, x0, sw_ctx, PETSC_FALSE));

  /* Initialize truth trajectory */
  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecCopy(x0, truth_state));
  PetscCall(VecDuplicate(x0, &rmse_work));

  /* Spinup if needed */
  if (n_spin > 0) {
    PetscInt spinup_progress_interval = (n_spin >= 10) ? (n_spin / 10) : 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinning up truth trajectory for %" PetscInt_FMT " steps...\n", n_spin));
    for (PetscInt k = 0; k < n_spin; k++) {
      PetscCall(ShallowWaterStep2DVec(sw_ctx, truth_time, truth_state));
      truth_time += dt;
      if ((k + 1) % spinup_progress_interval == 0 || (k + 1) == n_spin) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Spinup progress: %" PetscInt_FMT "/%" PetscInt_FMT "\n", k + 1, n_spin));
    }
    PetscCall(VecCopy(truth_state, x0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinup complete.\n\n"));
  }

  /* Create observation matrix H. Pass the DMDA-partitioned state local size so that
     H's column layout matches the state vector. */
  PetscCall(VecGetLocalSize(x0, &local_state_size));
  PetscCheck(local_state_size % ndof == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "local_state_size (%" PetscInt_FMT ") not a multiple of ndof (%" PetscInt_FMT ")", local_state_size, ndof);
  PetscCall(CreateObservationMatrix2D(nx, ny, ndof, obs_stride, local_state_size, &H, &H1, &nobs));

  /* Initialize observation vectors */
  PetscCall(MatCreateVecs(H, NULL, &observation));
  PetscCall(VecDuplicate(observation, &obs_noise));
  PetscCall(VecDuplicate(observation, &obs_error_var));
  PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

  /* Create and configure PetscDA for ensemble data assimilation */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(PetscDASetSizes(da, nx * ny * ndof, nobs));
  PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));
  PetscCall(VecGetLocalSize(observation, &local_obs_size));
  PetscCall(PetscDASetLocalSizes(da, local_state_size, local_obs_size));
  PetscCall(PetscDASetNDOF(da, ndof));
  PetscCall(PetscDASetFromOptions(da));
  PetscCall(PetscDAEnsembleGetSize(da, &ensemble_size));
  PetscCall(PetscDASetUp(da));

  /* Initialize ensemble statistics vectors */
  PetscCall(VecDuplicate(x0, &x_mean));
  PetscCall(VecDuplicate(x0, &x_forecast));

  /* Set observation error variance */
  PetscCall(PetscDASetObsErrorVariance(da, obs_error_var));

  /* Configure localization for LETKF. Built-in distance-based kernels (Gaspari-Cohn,
     Gaussian, boxcar) are wired through SetLocalizationCoordinates and the matrix Q
     is built lazily on the first analysis; the NONE kernel needs no setup. */
  PetscCall(ConfigureLETKFLocalization(da, da_state, nx, ny, ndof, local_state_size, Lx, Ly, H1, &localization_radius));

  /* Initialize ensemble members with perturbations */
  PetscCall(InitializeBalancedEnsemble(da, da_state, sw_ctx, random_seed, ensemble_size, init_perturb_amplitude, init_h_bias));

  /* Print configuration summary */
  dx  = Lx / nx;
  dy  = Ly / ny;
  c   = PetscSqrtReal(g * h0);
  cfl = dt * c * (1.0 / dx + 1.0 / dy);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2D Shallow Water LETKF Example\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==============================\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "  Mode                  : Data Assimilation\n"
                        "  Flux scheme           : Rusanov (1st order)\n"
                        "  Grid dimensions       : %" PetscInt_FMT " x %" PetscInt_FMT "\n"
                        "  State dimension       : %" PetscInt_FMT " (%" PetscInt_FMT " grid points x %" PetscInt_FMT " DOF)\n"
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
                        "  Init perturb amp      : %.3f\n"
                        "  Init height bias      : %.3f\n"
                        "  Observation noise std : %.3f\n"
                        "  Random seed           : %" PetscInt_FMT "\n",
                        nx, ny, nx * ny * ndof, nx * ny, ndof, nobs, obs_stride, ensemble_size, (double)Lx, (double)Ly, (double)dx, (double)dy, (double)h0, (double)Ax, (double)Ay, (double)g, (double)c, (double)dt, (double)cfl, steps, obs_freq, (double)init_perturb_amplitude, (double)init_h_bias, (double)obs_error_std, random_seed));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

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
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# nx=%" PetscInt_FMT ", ny=%" PetscInt_FMT ", ndof=%" PetscInt_FMT ", nobs=%" PetscInt_FMT ", ensemble_size=%" PetscInt_FMT "\n", nx, ny, ndof, nobs, ensemble_size));
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
    PetscCall(ShallowWaterStep2DVec(sw_ctx, truth_time, truth_state));
    truth_time += dt;

    /* Forecast step: compute ensemble mean and forecast RMSE */
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, nx * ny * ndof, &rmse_forecast));
    rmse_analysis = rmse_forecast;

    /* Analysis step: assimilate observations when available */
    if (step % obs_freq == 0) {
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
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "%" PetscInt_FMT " %.6f", step, (double)time));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(truth_array[i])));
      for (i = 0; i < nx * ny * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(mean_array[i])));
      if (step % obs_freq == 0) {
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
    if (step == steps || (progress_freq > 0 && step % progress_freq == 0))
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis));
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

  PetscCall(PetscDAView(da, PETSC_VIEWER_STDOUT_WORLD));

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
    requires: !complex
    args: -petscda_type letkf -steps 5 -progress_freq 1 -petscda_ensemble_size 10 -obs_freq 2 -obs_error 0.03 -nx 21 -ny 21

    test:
      suffix: letkf_wave2d
      args: -petscda_ensemble_size 7

    test:
      nsize: 3
      suffix: letkf_wave2d_mpi
      args: -petscda_ensemble_size 5 -petscda_letkf_localization_radius 10.0

    test:
      suffix: kokkos_wave2d_serial
      requires: kokkos_kernels !sycl
      args: -mat_type aijkokkos -vec_type kokkos -petscda_ensemble_size 7

    test:
      nsize: 3
      # oneapi::mkl::lapack::syevd: invalid argument: On entry, parameter 8 had an illegal value
      suffix: kokkos_wave2d
      requires: kokkos_kernels !sycl
      args: -mat_type aijkokkos -vec_type kokkos -petscda_ensemble_size 5 -petscda_letkf_localization_radius 10.0

    test:
      suffix: letkf_none
      args: -petscda_ensemble_size 7 -petscda_letkf_localization_type none

    test:
      suffix: letkf_gaussian
      args: -petscda_ensemble_size 7 -petscda_letkf_localization_type gaussian -petscda_letkf_localization_radius 10.0

    test:
      suffix: letkf_boxcar
      args: -petscda_ensemble_size 3 -petscda_letkf_localization_type boxcar -petscda_letkf_localization_radius 15.0

  # Exercises truth-trajectory spinup (-n_spin) and the visualization-data
  # writer (-output_file). Serial-only: the writer guards itself off under MPI.
  test:
    suffix: letkf_wave2d_spinup_io
    requires: !complex
    nsize: 1
    args: -petscda_type letkf -steps 3 -n_spin 2 -nx 11 -ny 11 -obs_freq 2 -obs_error 0.1 -petscda_ensemble_size 5 -petscda_letkf_localization_radius 10.0 -output_file ex4_spinup_io.dat
    temporaries: ex4_spinup_io.dat

  # Sparse + noisy observation regime (16x sparser, 10x noisier than the dense testset)
  # exercises the localized analysis path on a configuration where it is expected to
  # behave qualitatively differently from the unlocalized fast path.
  test:
    suffix: letkf_wave2d_sparse
    requires: !complex
    args: -steps 5 -progress_freq 1 -nx 21 -ny 21 -obs_freq 2 -obs_stride 8 -obs_error 0.3 -petscda_type letkf -petscda_ensemble_size 7 -petscda_letkf_localization_type gaspari_cohn -petscda_letkf_localization_radius 30.0
TEST*/
