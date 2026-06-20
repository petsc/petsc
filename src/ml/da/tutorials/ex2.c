static char help[] = "Deterministic LETKF example for the Lorenz-96 model. See "
                     "Algorithm 6.4 of \n"
                     "Asch, Bocquet, and Nodet (2016) \"Data Assimilation\" "
                     "(SIAM, doi:10.1137/1.9781611974546).\n\n"
                     "  Expected result: Similar to ETKF with full localization\n\n";

/* Data assimilation framework header (provides PetscDA) */
#include <petscda.h>
/* PETSc DMDA header (provides DM, DMDA functionality) */
#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>

/* Default parameter values */
#define DEFAULT_N             40
#define DEFAULT_STEPS         105000
#define DEFAULT_BURN          5000
#define DEFAULT_OBS_FREQ      5
#define DEFAULT_RANDOM_SEED   12345
#define DEFAULT_F             8.0
#define DEFAULT_DT            0.05
#define DEFAULT_OBS_ERROR_STD 1.0
#define DEFAULT_ENSEMBLE_SIZE 30
#define SPINUP_STEPS          300 /* Spin up truth to Lorenz-96 attractor (~200 steps sufficient, 300 for safety) */

/* Minimum valid parameter values */
#define MIN_N              1
#define MIN_ENSEMBLE_SIZE  2
#define MIN_OBS_FREQ       1
#define PROGRESS_INTERVALS 10

typedef struct {
  DM        da; /* 1D periodic DM storing the Lorenz-96 state */
  PetscInt  n;  /* State dimension (number of grid points) */
  PetscReal F;  /* Constant forcing term in the Lorenz-96 equations */
  PetscReal dt; /* Integration time step size */
  TS        ts; /* Reusable time stepper for efficiency */
} Lorenz96Ctx;

/*
  Lorenz96RHS - Compute the right-hand side of the Lorenz-96 equations
*/
static PetscErrorCode Lorenz96RHS(TS ts, PetscReal t, Vec X, Vec F_vec, PetscCtx ctx)
{
  Lorenz96Ctx       *l95 = (Lorenz96Ctx *)ctx;
  Vec                X_local;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt           xs, xm, i;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(l95->da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMGetLocalVector(l95->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecGetArray(l95->da, F_vec, &f));

  for (i = xs; i < xs + xm; i++) f[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + l95->F;

  PetscCall(DMDAVecRestoreArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecRestoreArray(l95->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(l95->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96ContextCreate - Create and initialize a Lorenz96 context with reusable TS object
*/
static PetscErrorCode Lorenz96ContextCreate(DM da, PetscInt n, PetscReal F, PetscReal dt, Lorenz96Ctx **ctx)
{
  Lorenz96Ctx *l95;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&l95));
  l95->da = da;
  l95->n  = n;
  l95->F  = F;
  l95->dt = dt;

  PetscCall(TSCreate(PetscObjectComm((PetscObject)da), &l95->ts));
  PetscCall(TSSetProblemType(l95->ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(l95->ts, NULL, Lorenz96RHS, l95));
  PetscCall(TSSetType(l95->ts, TSRK));
  PetscCall(TSRKSetType(l95->ts, TSRK4));
  PetscCall(TSSetTimeStep(l95->ts, dt));
  PetscCall(TSSetMaxSteps(l95->ts, 1));
  PetscCall(TSSetMaxTime(l95->ts, dt));
  PetscCall(TSSetExactFinalTime(l95->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(l95->ts));

  *ctx = l95;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96ContextDestroy - Destroy a Lorenz96 context and its TS object
*/
static PetscErrorCode Lorenz96ContextDestroy(Lorenz96Ctx **ctx)
{
  PetscFunctionBeginUser;
  if (!ctx || !*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Advance a single state vector one TS step. Used by the truth trajectory and as the per-column kernel of Lorenz96Step(). */
static PetscErrorCode Lorenz96StepVec(Lorenz96Ctx *l95, Vec x)
{
  PetscFunctionBeginUser;
  PetscCall(TSSetStepNumber(l95->ts, 0));
  PetscCall(TSSetTime(l95->ts, 0.0));
  PetscCall(TSSolve(l95->ts, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96Step - Advance every column of the ensemble matrix one time step using Lorenz-96 dynamics.
  TS only advances one state at a time, so loop over columns here.
*/
static PetscErrorCode Lorenz96Step(Mat ensemble, PetscCtx ctx)
{
  Lorenz96Ctx *l95 = (Lorenz96Ctx *)ctx;
  PetscInt     n;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(ensemble, NULL, &n));
  for (PetscInt j = 0; j < n; j++) {
    Vec col;

    PetscCall(MatDenseGetColumnVec(ensemble, j, &col));
    PetscCall(Lorenz96StepVec(l95, col));
    PetscCall(MatDenseRestoreColumnVec(ensemble, j, &col));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateIdentityObservationMatrix - Create identity observation matrix H for Lorenz-96

  For the fully observed case, H is an nxn identity matrix representing y = H*x where
  each observation corresponds directly to a state variable.

  Input Parameter:
. n - State dimension (number of grid points)

  Output Parameter:
. H - Identity observation matrix (n x n), sparse AIJ format (H in P x N)
*/
static PetscErrorCode CreateIdentityObservationMatrix(PetscInt n, Mat *H)
{
  PetscInt i;

  PetscFunctionBeginUser;
  /* Create identity observation matrix H (n x n) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 0, NULL, H));
  PetscCall(MatSetFromOptions(*H));

  /* Set diagonal entries to 1.0 for identity mapping */
  for (i = 0; i < n; i++) PetscCall(MatSetValue(*H, i, i, 1.0, INSERT_VALUES));

  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ValidateParameters - Validate input parameters and apply constraints
*/
static PetscErrorCode ValidateParameters(PetscInt *n, PetscInt *steps, PetscInt *burn, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *F, PetscReal *obs_error_std)
{
  PetscFunctionBeginUser;
  PetscCheck(*n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "State dimension n must be positive, got %" PetscInt_FMT, *n);
  PetscCheck(*steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative, got %" PetscInt_FMT, *steps);
  PetscCheck(*ensemble_size >= MIN_ENSEMBLE_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least %" PetscInt_FMT " for meaningful statistics, got %" PetscInt_FMT, (PetscInt)MIN_ENSEMBLE_SIZE, *ensemble_size);

  if (*obs_freq < MIN_OBS_FREQ) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency adjusted from %" PetscInt_FMT " to %" PetscInt_FMT "\n", *obs_freq, (PetscInt)MIN_OBS_FREQ));
    *obs_freq = MIN_OBS_FREQ;
  }
  if (*obs_freq > *steps && *steps > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency (%" PetscInt_FMT ") > total steps (%" PetscInt_FMT "), no observations will be assimilated.\n", *obs_freq, *steps));
  if (*burn > *steps) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Burn-in steps (%" PetscInt_FMT ") exceeds total steps (%" PetscInt_FMT "), setting burn = steps\n", *burn, *steps));
    *burn = *steps;
  }

  PetscCheck(*dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step dt must be positive, got %g", (double)*dt);
  PetscCheck(*obs_error_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observation error std must be positive, got %g", (double)*obs_error_std);
  PetscCheck(PetscIsNormalReal(*F), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Forcing parameter F must be a normal real number");
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

int main(int argc, char **argv)
{
  Lorenz96Ctx                 *l95_ctx = NULL, *truth_ctx = NULL;
  DM                           da_state;
  PetscDA                      da;
  Vec                          x0, x_mean, x_forecast;
  Vec                          truth_state, rmse_work;
  Vec                          observation, obs_noise, obs_error_var;
  Mat                          H = NULL; /* Observation operator matrix */
  PetscRandom                  rng;
  Vec                          xyz[3] = {NULL, NULL, NULL};
  Vec                          coord;
  PetscDALETKFLocalizationType loc_type;
  PetscBool                    radius_set;
  const char                  *da_prefix;
  PetscInt                     n = DEFAULT_N, steps = DEFAULT_STEPS, burn = DEFAULT_BURN, obs_freq = DEFAULT_OBS_FREQ;
  PetscInt                     random_seed = DEFAULT_RANDOM_SEED, ensemble_size = DEFAULT_ENSEMBLE_SIZE;
  PetscInt                     n_stat_steps = 0, n_obs_stat_steps = 0, obs_count = 0, step, progress_interval;
  PetscReal                    F = DEFAULT_F, dt = DEFAULT_DT, obs_error_std = DEFAULT_OBS_ERROR_STD;
  PetscReal                    ensemble_init_std = -1; /* Initial ensemble spread */
  PetscReal                    localization_radius;    /* Default 2*domain: effectively no localization with Gaspari-Cohn (max periodic distance is L/2) */
  PetscReal                    bd[3]         = {DEFAULT_N, 0, 0};
  PetscReal                    rmse_forecast = 0.0, rmse_analysis = 0.0, spread = 0.0;
  PetscReal                    sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Kokkos initialization deferred to Phase 5 optimization */

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Lorenz-96 Example", NULL);
  PetscCall(PetscOptionsInt("-n", "State dimension", "", n, &n, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsInt("-burn", "Burn-in steps excluded from statistics", "", burn, &burn, NULL));
  PetscCall(PetscOptionsInt("-obs_freq", "Observation frequency", "", obs_freq, &obs_freq, NULL));
  PetscCall(PetscOptionsReal("-F", "Forcing parameter", "", F, &F, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation error standard deviation", "", obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsReal("-ensemble_init_std", "Initial ensemble spread standard deviation", "", ensemble_init_std, &ensemble_init_std, NULL));
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble perturbations", "", random_seed, &random_seed, NULL));
  PetscOptionsEnd();
  bd[0]               = (PetscReal)n;
  localization_radius = 2.0 * bd[0];

  if (ensemble_init_std < 0) ensemble_init_std = obs_error_std;

  /* Validate and constrain parameters */
  PetscCall(ValidateParameters(&n, &steps, &burn, &obs_freq, &ensemble_size, &dt, &F, &obs_error_std));

  /* Calculate progress reporting interval */
  progress_interval = (steps >= PROGRESS_INTERVALS) ? (steps / PROGRESS_INTERVALS) : 1;

  /* Create 1D periodic DM for state space */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, n, 1, 2, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));
  PetscCall(DMDASetUniformCoordinates(da_state, 0.0, (PetscReal)n, 0.0, 0.0, 0.0, 0.0));

  /* Create Lorenz96 context with reusable TS object */
  PetscCall(Lorenz96ContextCreate(da_state, n, F, dt, &l95_ctx));
  PetscCall(Lorenz96ContextCreate(da_state, n, F, dt, &truth_ctx));

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
  PetscCall(PetscRandomSetInterval(rng, -.1 * F, .1 * F));
  PetscCall(VecSetRandom(x0, rng));
  PetscCall(PetscRandomSetInterval(rng, 0, 1));

  /* Initialize truth trajectory */
  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecCopy(x0, truth_state));
  PetscCall(VecDuplicate(x0, &rmse_work));

  /* Spin up truth to get onto attractor */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinning up truth for %" PetscInt_FMT " steps...\n", (PetscInt)SPINUP_STEPS));
  for (PetscInt k = 0; k < SPINUP_STEPS; k++) PetscCall(Lorenz96StepVec(truth_ctx, truth_state));

  /* Initialize observation vectors */
  PetscCall(VecDuplicate(x0, &observation));
  PetscCall(VecDuplicate(x0, &obs_noise));
  PetscCall(VecDuplicate(x0, &obs_error_var));
  PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

  /* Initialize ensemble statistics vectors */
  PetscCall(VecDuplicate(x0, &x_mean));
  PetscCall(VecDuplicate(x0, &x_forecast));

  /* Create identity observation matrix H */
  PetscCall(CreateIdentityObservationMatrix(n, &H));

  /* Create and configure PetscDA for ensemble data assimilation */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
  /* Note: ndof defaults to 1 (scalar field) - perfect for Lorenz-96 */
  PetscCall(PetscDASetSizes(da, n, n));
  PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));
  PetscCall(PetscDASetFromOptions(da));
  PetscCall(PetscDAEnsembleGetSize(da, &ensemble_size));
  PetscCall(PetscDASetUp(da));
  PetscCall(PetscDASetObsErrorVariance(da, obs_error_var));

  /* Configure localization for LETKF (Q is built lazily on first analysis). PETSCDALETKF is
     now the only registered PetscDAType, so we can call the setters unconditionally. */
  PetscCall(DMGetCoordinates(da_state, &coord));
  PetscCall(DMCreateGlobalVector(da_state, &xyz[0]));
  PetscCall(PetscObjectSetName((PetscObject)xyz[0], "x_coordinate"));
  PetscCall(VecStrideGather(coord, 0, xyz[0], INSERT_VALUES));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)da, &da_prefix));
  PetscCall(PetscOptionsHasName(NULL, da_prefix, "-petscda_letkf_localization_radius", &radius_set));
  if (!radius_set) PetscCall(PetscDALETKFSetLocalizationRadius(da, localization_radius));
  PetscCall(PetscDALETKFGetLocalizationRadius(da, &localization_radius));
  PetscCall(PetscDALETKFSetLocalizationCoordinates(da, xyz, bd, H));
  PetscCall(VecDestroy(&xyz[0]));
  PetscCall(PetscDALETKFGetLocalizationType(da, &loc_type));
  if (loc_type != PETSCDA_LETKF_LOC_NONE) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization configured: %" PetscInt_FMT " vertices, radius=%g\n", n, (double)localization_radius));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization disabled (LETKF NONE; equivalent to global ETKF)\n"));

  /* Initialize ensemble members from spun-up truth state */
  PetscCall(PetscDAEnsembleInitialize(da, truth_state, ensemble_init_std, rng));

  /* Print configuration summary */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Lorenz-96 LETKF Example\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "  State dimension        : %" PetscInt_FMT "\n"
                        "  Ensemble size          : %" PetscInt_FMT "\n"
                        "  Forcing parameter (F)  : %.4f\n"
                        "  Time step (dt)         : %.4f\n"
                        "  Total steps            : %" PetscInt_FMT "\n"
                        "  Burn-in steps          : %" PetscInt_FMT "\n"
                        "  Spin-up steps          : %" PetscInt_FMT "\n"
                        "  Observation frequency  : %" PetscInt_FMT "\n"
                        "  Observation noise std  : %.3f\n"
                        "  Ensemble init std      : %.3f\n"
                        "  Random seed            : %" PetscInt_FMT "\n",
                        n, ensemble_size, (double)F, (double)dt, steps, burn, (PetscInt)SPINUP_STEPS, obs_freq, (double)obs_error_std, (double)ensemble_init_std, random_seed));
  if (loc_type != PETSCDA_LETKF_LOC_NONE) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Localization radius    : %g\n", (double)localization_radius));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  /* Main assimilation cycle: forecast and analysis steps */
  for (step = 0; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Forecast step: compute ensemble mean and forecast RMSE */
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, n, &rmse_forecast));
    rmse_analysis = rmse_forecast;

    /* Analysis step: assimilate observations when available */
    if (step % obs_freq == 0 && step > 0) {
      /* Generate synthetic noisy observations from truth */
      PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
      PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_state));

      /* Perform LETKF analysis with observation matrix H */
      PetscCall(PetscDAEnsembleAnalysis(da, observation, H));

      /* Compute analysis RMSE */
      PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
      PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, n, &rmse_analysis));
      obs_count++;
    }

    /* Accumulate statistics after burn-in period.
       Forecast RMSE is accumulated every step; analysis RMSE only at observation times
       to avoid conflating forecast and analysis errors. */
    if (step >= burn) {
      sum_rmse_forecast += rmse_forecast;
      n_stat_steps++;
      if (step % obs_freq == 0 && step > 0) {
        sum_rmse_analysis += rmse_analysis;
        n_obs_stat_steps++;
      }
    }

    /* Progress reporting */
    if ((step % progress_interval == 0) || (step == steps) || (step == 0)) {
      Mat       X_anom;
      PetscReal norm_fro;

      PetscCall(PetscDAEnsembleComputeAnomalies(da, x_mean, &X_anom));
      PetscCall(MatNorm(X_anom, NORM_FROBENIUS, &norm_fro));
      spread = norm_fro / PetscSqrtReal((PetscReal)n);
      PetscCall(MatDestroy(&X_anom));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f  Spread %.5f%s\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis, (double)spread, (step < burn) ? " [burn-in]" : ""));
    }

    /* Propagate ensemble and truth trajectory */
    if (step < steps) {
      PetscCall(PetscDAEnsembleForecast(da, Lorenz96Step, l95_ctx));
      PetscCall(Lorenz96StepVec(truth_ctx, truth_state));
    }
  }

  /* Report final statistics */
  if (n_stat_steps > 0) {
    PetscReal avg_rmse_forecast = sum_rmse_forecast / n_stat_steps;
    PetscReal avg_rmse_analysis = (n_obs_stat_steps > 0) ? sum_rmse_analysis / n_obs_stat_steps : 0.0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStatistics (%" PetscInt_FMT " forecast steps, %" PetscInt_FMT " analysis steps post-burn-in):\n", n_stat_steps, n_obs_stat_steps));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==================================================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (forecast) : %.5f\n", (double)avg_rmse_forecast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (analysis) : %.5f\n", (double)avg_rmse_analysis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Observations used    : %" PetscInt_FMT "\n\n", obs_count));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nWarning: No post-burn-in statistics collected (burn >= steps)\n\n"));
  }

  PetscCall(PetscDAView(da, PETSC_VIEWER_STDOUT_WORLD));

  /* Cleanup */
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&x_forecast));
  PetscCall(VecDestroy(&x_mean));
  PetscCall(VecDestroy(&obs_error_var));
  PetscCall(VecDestroy(&obs_noise));
  PetscCall(VecDestroy(&observation));
  PetscCall(VecDestroy(&rmse_work));
  PetscCall(VecDestroy(&truth_state));
  PetscCall(VecDestroy(&x0));
  PetscCall(PetscDADestroy(&da));
  PetscCall(DMDestroy(&da_state));
  PetscCall(Lorenz96ContextDestroy(&l95_ctx));
  PetscCall(Lorenz96ContextDestroy(&truth_ctx));
  PetscCall(PetscRandomDestroy(&rng));

  /* Kokkos finalization deferred to Phase 5 optimization */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: !complex
    args: -steps 20 -burn 5 -obs_freq 1 -obs_error 1 -petscda_ensemble_size 5 -petscda_type letkf

    test:
      suffix: letkf_serial

    test:
      suffix: letkf_loc_none
      args: -petscda_letkf_localization_type none

    test:
      nsize: 2
      suffix: letkf_cpu_2rank
      args: -petscda_letkf_localization_radius 5.0

    test:
      nsize: 2
      suffix: letkf_loc_none_2rank
      args: -petscda_letkf_localization_type none

  testset:
    requires: kokkos_kernels !complex !sycl
    args: -steps 20 -burn 5 -obs_freq 1 -obs_error 1 -petscda_ensemble_size 5 -petscda_type letkf -mat_type aijkokkos -dm_vec_type kokkos

    test:
      nsize: 3
      suffix: letkf
      # oneapi::mkl::lapack::syevd: invalid argument: On entry, parameter 8 had an illegal value
      args: -info :vec -petscda_letkf_localization_radius 5.0

    test:
      suffix: letkf_loc_none_kokkos
      args: -petscda_letkf_localization_type none

    test:
      nsize: 3
      suffix: letkf_loc_none_kokkos_3rank
      args: -petscda_letkf_localization_type none

  TEST*/
