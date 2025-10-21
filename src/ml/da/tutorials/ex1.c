/* Data assimilation framework header (provides PetscDA) */
#include <petscda.h>
/* PETSc DMDA header (provides DM, DMDA functionality) */
#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>

static char help[] = "Deterministic ETKF example for the Lorenz-96 model. See "
                     "Algorithm 6.4 of \n"
                     "Asch, Bocquet, and Nodet (2016) \"Data Assimilation\" "
                     "(SIAM, doi:10.1137/1.9781611974546).\n\n"
                     "  Note: Asch et al. run Lorenz-96 for 100,000 steps, but our ensemble collapses.\n\n";

/* \begin{algorithm}
\caption{Ensemble Transform Kalman Filter (ETKF) - Deterministic}
\begin{algorithmic}[1]
\State \textbf{Initialize:} Ensemble $\mathbf{E}_0 = [\mathbf{x}_0^{(1)}, \ldots, \mathbf{x}_0^{(m)}]$ \Comment{$\mathbf{E}_0 \in \mathbb{R}^{n \times m}$}
\For{$k = 1, 2, \ldots, K$}
    \State \textbf{// Forecast Step}
    \For{$i = 1$ to $m$}
        \State $\mathbf{x}_k^{f,(i)} = \mathcal{M}_{k-1}(\mathbf{x}_{k-1}^{a,(i)})$ \Comment{$\mathbf{x}_k^{f,(i)} \in \mathbb{R}^{n}$}
    \EndFor
    \State Assemble forecast matrix: $\mathbf{E}_k^f = [\mathbf{x}_k^{f,(1)}, \ldots, \mathbf{x}_k^{f,(m)}]$ \Comment{$\mathbf{E}_k^f \in \mathbb{R}^{n \times m}$}
    \State
    \State \textbf{// Analysis Step (if observation available)}
    \If{observation $\mathbf{y}_k$ available}
        \State Compute ensemble mean: $\overline{\mathbf{x}}_k^f = \frac{1}{m}\sum_{i=1}^m \mathbf{x}_k^{f,(i)}$ \Comment{$\overline{\mathbf{x}}_k^f \in \mathbb{R}^{n}$}
        \State Compute anomalies: $\mathbf{X}_k = \frac{1}{\sqrt{m-1}}(\mathbf{E}_k^f - \overline{\mathbf{x}}_k^f \mathbf{1}^T)$ \Comment{$\mathbf{X}_k \in \mathbb{R}^{n \times m}$}
        \State Apply observation operator: $\mathbf{Z}_k = \mathcal{H}(\mathbf{E}_k^f)$ \Comment{$\mathbf{Z}_k \in \mathbb{R}^{b \times m}$}
        \State Compute obs ensemble mean: $\overline{\mathbf{y}}_k = \frac{1}{m}\sum_{i=1}^m \mathcal{H}(\mathbf{x}_k^{f,(i)})$ \Comment{$\overline{\mathbf{y}}_k \in \mathbb{R}^{b}$}
        \State Compute obs anomalies: $\mathbf{S}_k = \frac{1}{\sqrt{m-1}}\mathbf{R}^{-1/2}(\mathbf{Z}_k - \overline{\mathbf{y}}_k\mathbf{1}^T)$ \Comment{$\mathbf{S}_k \in \mathbb{R}^{b \times m}$}
        \State Compute raw innovation: $\boldsymbol{\delta}_k = \mathbf{y}_k - \overline{\mathbf{y}}_k$ \Comment{$\boldsymbol{\delta}_k \in \mathbb{R}^{b}$}
        \State Whiten innovation: $\tilde{\boldsymbol{\delta}}_k = \mathbf{R}^{-1/2}\boldsymbol{\delta}_k$ \Comment{$\tilde{\boldsymbol{\delta}}_k \in \mathbb{R}^{b}$}
        \State Compute transform matrix: $\mathbf{T}_k = (\mathbf{I}_m + \mathbf{S}_k^T\mathbf{S}_k)^{-1}$ \Comment{$\mathbf{T}_k \in \mathbb{R}^{m \times m}$}
        \State Compute weight vector: $\mathbf{w}_k = \mathbf{T}_k \mathbf{S}_k^T \tilde{\boldsymbol{\delta}}_k$ \Comment{$\mathbf{w}_k \in \mathbb{R}^{m}$}
        \State Form deterministic map: $\mathbf{G}_k = \mathbf{w}_k\mathbf{1}^T + \sqrt{m-1}\,\mathbf{T}_k^{1/2}$ \Comment{$\mathbf{G}_k \in \mathbb{R}^{m \times m}$, $\mathbf{U} = \mathbf{I}_m$}
        \State Update ensemble: $\mathbf{E}_k^a = \overline{\mathbf{x}}_k^f\mathbf{1}^T + \mathbf{X}_k \mathbf{G}_k$ \Comment{$\mathbf{E}_k^a \in \mathbb{R}^{n \times m}$}
    \Else
        \State $\mathbf{E}_k^a = \mathbf{E}_k^f$ \Comment{$\mathbf{E}_k^a \in \mathbb{R}^{n \times m}$}
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}
 */

/* Default parameter values */
#define DEFAULT_N             40
#define DEFAULT_STEPS         105000
#define DEFAULT_BURN          5000
#define DEFAULT_OBS_FREQ      1
#define DEFAULT_RANDOM_SEED   12345
#define DEFAULT_F             8.0
#define DEFAULT_DT            0.05
#define DEFAULT_OBS_ERROR_STD 1.0
#define DEFAULT_ENSEMBLE_SIZE 30
#define SPINUP_STEPS          0

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

  Input Parameters:
+ ts    - The time-stepping context (unused but required by interface)
. t     - Current time (unused but required by interface)
. X     - State vector
- ctx   - User context (Lorenz96Ctx)

  Output Parameter:
. F_vec - RHS vector (tendency)
*/
static PetscErrorCode Lorenz96RHS(TS ts, PetscReal t, Vec X, Vec F_vec, PetscCtx ctx)
{
  Lorenz96Ctx       *l95 = (Lorenz96Ctx *)ctx;
  Vec                X_local;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt           xs, xm, i;

  PetscFunctionBeginUser;
  (void)ts; /* Mark as intentionally unused to avoid compiler warnings */
  (void)t;

  /* Work with a local (ghosted) vector so the Lorenz-96 stencil has the
   * required neighbors for periodic boundary conditions. */
  PetscCall(DMDAGetCorners(l95->da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMGetLocalVector(l95->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecGetArray(l95->da, F_vec, &f));

  /* Standard Lorenz-96 tendency: (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F. */
  for (i = xs; i < xs + xm; i++) f[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + l95->F;

  PetscCall(DMDAVecRestoreArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecRestoreArray(l95->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(l95->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96ContextCreate - Create and initialize a Lorenz96 context with reusable TS object

  Input Parameters:
+ da - DM for state space
. n  - State dimension
. F  - Forcing parameter
. dt - Time step size

  Output Parameter:
. ctx - Initialized Lorenz96 context
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

  /* Create and configure a reusable time stepper to avoid repeated allocation/deallocation */
  PetscCall(TSCreate(PetscObjectComm((PetscObject)da), &l95->ts));
  PetscCall(TSSetProblemType(l95->ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(l95->ts, NULL, Lorenz96RHS, l95));
  PetscCall(TSSetType(l95->ts, TSRK));
  PetscCall(TSRKSetType(l95->ts, TSRK4));
  PetscCall(TSSetTimeStep(l95->ts, dt));
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

/*
  Lorenz96Step - Advance state vector one time step using Lorenz-96 dynamics

  Input Parameters:
+ input - state vector to be advanced
- ctx   - Lorenz96 context (contains reusable TS)

  Output Parameter:
. output - state vector after one time step

  Notes:
  Uses a single explicit RK4 step with the pre-configured TS object for efficiency.
*/
static PetscErrorCode Lorenz96Step(Vec input, Vec output, PetscCtx ctx)
{
  Lorenz96Ctx *l95 = (Lorenz96Ctx *)ctx;

  PetscFunctionBeginUser;
  /* Reset the TS time for each integration */
  PetscCall(TSSetTime(l95->ts, 0.0));
  if (input != output) PetscCall(VecCopy(input, output));
  PetscCall(TSSolve(l95->ts, output));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateIdentityObservationMatrix - Create identity observation matrix H for Lorenz-96

  For the fully observed case, H is an nxn identity matrix representing y = H*x where
  each observation corresponds directly to a state variable.

  Input Parameter:
. n - State dimension (number of grid points)

  Output Parameter:
. H - Identity observation matrix (n x n), sparse AIJ format
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

  Input/Output Parameters:
+ n             - State dimension
. steps         - Number of time steps
. burn          - Burn-in steps
. obs_freq      - Observation frequency
. ensemble_size - Ensemble size
. dt            - Time step
. F             - Forcing parameter
- obs_error_std - Observation error standard deviation

  This routine should be reduced/eliminated. It is the job of PETSc functions to validate input, not the tutorials code
*/
static PetscErrorCode ValidateParameters(PetscInt *n, PetscInt *steps, PetscInt *burn, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *F, PetscReal *obs_error_std)
{
  PetscFunctionBeginUser;
  /* Validate and constrain integer parameters */
  PetscCheck(*n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "State dimension n must be positive, got %" PetscInt_FMT, *n);
  PetscCheck(*steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative, got %" PetscInt_FMT, *steps);
  PetscCheck(*ensemble_size >= MIN_ENSEMBLE_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least %" PetscInt_FMT " for meaningful statistics, got %" PetscInt_FMT, (PetscInt)MIN_ENSEMBLE_SIZE, *ensemble_size);

  /* Apply constraints */
  if (*obs_freq < MIN_OBS_FREQ) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency adjusted from %" PetscInt_FMT " to %" PetscInt_FMT "\n", *obs_freq, (PetscInt)MIN_OBS_FREQ));
    *obs_freq = MIN_OBS_FREQ;
  }
  if (*obs_freq > *steps && *steps > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency (%" PetscInt_FMT ") > total steps (%" PetscInt_FMT "), no observations will be assimilated.\n", *obs_freq, *steps));
  if (*burn > *steps) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Burn-in steps (%" PetscInt_FMT ") exceeds total steps (%" PetscInt_FMT "), setting burn = steps\n", *burn, *steps));
    *burn = *steps;
  }

  /* Validate real-valued parameters */
  PetscCheck(*dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step dt must be positive, got %g", (double)*dt);
  PetscCheck(*obs_error_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observation error std must be positive, got %g", (double)*obs_error_std);
  PetscCheck(PetscIsNormalReal(*F), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Forcing parameter F must be a normal real number");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeRMSE - Compute root mean square error between two vectors

  Input Parameters:
+ v1 - First vector
. v2 - Second vector
- n  - Vector dimension (for normalization)

  Output Parameter:
. rmse - Root mean square error
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
  /* Configuration parameters */
  PetscInt  n                 = DEFAULT_N;
  PetscInt  steps             = DEFAULT_STEPS;
  PetscInt  burn              = DEFAULT_BURN;
  PetscInt  obs_freq          = DEFAULT_OBS_FREQ;
  PetscInt  random_seed       = DEFAULT_RANDOM_SEED;
  PetscInt  ensemble_size     = DEFAULT_ENSEMBLE_SIZE;
  PetscReal F                 = DEFAULT_F;
  PetscReal dt                = DEFAULT_DT;
  PetscReal obs_error_std     = DEFAULT_OBS_ERROR_STD;
  PetscReal ensemble_init_std = 1; /* Initial ensemble spread */

  /* PETSc objects */
  Lorenz96Ctx *l95_ctx = NULL, *truth_ctx = NULL;
  DM           da_state;
  PetscDA      da;
  Vec          x0, x_mean, x_forecast;
  Vec          truth_state, rmse_work;
  Vec          observation, obs_noise, obs_error_var;
  PetscRandom  rng;
  Mat          H = NULL; /* Observation operator matrix */

  /* Statistics tracking */
  PetscReal rmse_forecast = 0.0, rmse_analysis = 0.0;
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt  n_stat_steps = 0;
  PetscInt  obs_count    = 0;
  PetscInt  step, progress_interval;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Lorenz-96 ETKF Quick Example", NULL);
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

  /* Validate and constrain parameters */
  PetscCall(ValidateParameters(&n, &steps, &burn, &obs_freq, &ensemble_size, &dt, &F, &obs_error_std));

  /* Calculate progress reporting interval (avoid division by zero) */
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
  PetscCall(PetscRandomSetSeed(rng, (unsigned long)random_seed));
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
  for (PetscInt k = 0; k < SPINUP_STEPS; k++) PetscCall(Lorenz96Step(truth_state, truth_state, truth_ctx));

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
  PetscCall(PetscDASetType(da, PETSCDAETKF)); /* Set ETKF type */
  PetscCall(PetscDASetSizes(da, n, n));
  PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));
  PetscCall(PetscDASetFromOptions(da));
  PetscCall(PetscDAEnsembleGetSize(da, &ensemble_size));
  PetscCall(PetscDASetUp(da));
  PetscCall(PetscDAViewFromOptions(da, NULL, "-petscda_view"));
  PetscCall(PetscDASetObsErrorVariance(da, obs_error_var));

  /* Initialize ensemble members from spun-up truth state with appropriate spread */
  PetscCall(PetscDAEnsembleInitialize(da, truth_state, ensemble_init_std, rng));

  /* Print configuration summary */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Lorenz-96 ETKF Example\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "  State dimension       : %" PetscInt_FMT "\n"
                        "  Ensemble size         : %" PetscInt_FMT "\n"
                        "  Forcing parameter (F) : %.4f\n"
                        "  Time step (dt)        : %.4f\n"
                        "  Total steps           : %" PetscInt_FMT "\n"
                        "  Burn-in steps         : %" PetscInt_FMT "\n"
                        "  Observation frequency : %" PetscInt_FMT "\n"
                        "  Observation noise std : %.3f\n"
                        "  Ensemble init std     : %.3f\n"
                        "  Random seed           : %" PetscInt_FMT "\n\n",
                        n, ensemble_size, (double)F, (double)dt, steps, burn, obs_freq, (double)obs_error_std, (double)ensemble_init_std, random_seed));

  /* Main assimilation cycle: forecast and analysis steps */
  for (step = 0; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Forecast step: compute ensemble mean and forecast RMSE */
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, n, &rmse_forecast));
    rmse_analysis = rmse_forecast; /* Default to forecast RMSE if no analysis */

    /* Analysis step: assimilate observations when available */
    if (step % obs_freq == 0 && step > 0) {
      /* Generate synthetic noisy observations from truth */
      PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
      PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_state));

      /* Perform ETKF analysis with observation matrix H */
      PetscCall(PetscDAEnsembleAnalysis(da, observation, H));

      /* Compute analysis RMSE */
      PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
      PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, n, &rmse_analysis));
      obs_count++;
    }

    /* Accumulate statistics after burn-in period */
    if (step >= burn) {
      sum_rmse_forecast += rmse_forecast;
      sum_rmse_analysis += rmse_analysis;
      n_stat_steps++;
    }

    /* Progress reporting */
    if ((step % progress_interval == 0) || (step == steps) || (step == 0)) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f%s\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis, (step < burn) ? " [burn-in]" : ""));
    }

    /* Propagate ensemble and truth trajectory */
    if (step < steps) {
      PetscCall(PetscDAEnsembleForecast(da, Lorenz96Step, l95_ctx));
      PetscCall(Lorenz96Step(truth_state, truth_state, truth_ctx));
    }
  }

  /* Report final statistics */
  if (n_stat_steps > 0) {
    PetscReal avg_rmse_forecast = sum_rmse_forecast / n_stat_steps;
    PetscReal avg_rmse_analysis = sum_rmse_analysis / n_stat_steps;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStatistics (%" PetscInt_FMT " post-burn-in steps):\n", n_stat_steps));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==================================================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (forecast) : %.5f\n", (double)avg_rmse_forecast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (analysis) : %.5f\n", (double)avg_rmse_analysis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Observations used    : %" PetscInt_FMT "\n\n", obs_count));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nWarning: No post-burn-in statistics collected (burn >= steps)\n\n"));
  }

  /* Test VecSetRandomGaussian to verify Gaussian distribution */
  {
    Vec          test_vec;
    PetscInt     test_size = 10000; /* Large sample for statistical testing */
    PetscScalar *array;
    PetscReal    mean_target = 2.0, std_target = 1.5;
    PetscReal    sample_mean = 0.0, sample_variance = 0.0, sample_std;
    PetscReal    skewness = 0.0, kurtosis = 0.0;
    PetscInt     i;
    PetscBool    test_gaussian = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_gaussian", &test_gaussian, NULL));

    if (test_gaussian) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n==============================================\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing VecSetRandomGaussian\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==============================================\n"));

      /* Create test vector */
      PetscCall(VecCreate(PETSC_COMM_WORLD, &test_vec));
      PetscCall(VecSetSizes(test_vec, PETSC_DECIDE, test_size));
      PetscCall(VecSetFromOptions(test_vec));

      /* Generate Gaussian random numbers */
      PetscCall(VecSetRandomGaussian(test_vec, rng, mean_target, std_target));

      /* Get array for statistical analysis */
      PetscCall(VecGetArray(test_vec, &array));

      /* Compute sample mean */
      for (i = 0; i < test_size; i++) sample_mean += PetscRealPart(array[i]);
      sample_mean /= test_size;

      /* Compute sample variance and higher moments */
      for (i = 0; i < test_size; i++) {
        PetscReal diff  = PetscRealPart(array[i]) - sample_mean;
        PetscReal diff2 = diff * diff;
        sample_variance += diff2;
        skewness += diff * diff2;
        kurtosis += diff2 * diff2;
      }
      sample_variance /= (test_size - 1);
      sample_std = PetscSqrtReal(sample_variance);

      /* Normalize skewness and kurtosis */
      skewness = (skewness / test_size) / PetscPowReal(sample_std, 3.0);
      kurtosis = (kurtosis / test_size) / PetscPowReal(sample_std, 4.0) - 3.0; /* Excess kurtosis */

      PetscCall(VecRestoreArray(test_vec, &array));

      /* Report results */
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTarget parameters:\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean      : %.6f\n", (double)mean_target));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Std Dev   : %.6f\n", (double)std_target));

      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nSample statistics (n=%" PetscInt_FMT "):\n", (PetscInt)test_size));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean      : %.6f (error: %.6f)\n", (double)sample_mean, (double)PetscAbsReal(sample_mean - mean_target)));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Std Dev   : %.6f (error: %.6f)\n", (double)sample_std, (double)PetscAbsReal(sample_std - std_target)));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Skewness  : %.6f (expected ~0 for Gaussian)\n", (double)skewness));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Kurtosis  : %.6f (expected ~0 for Gaussian)\n", (double)kurtosis));

      /* Statistical tests with reasonable tolerances for finite samples */
      PetscReal mean_error     = PetscAbsReal(sample_mean - mean_target);
      PetscReal std_error      = PetscAbsReal(sample_std - std_target);
      PetscReal mean_tolerance = 3.0 * std_target / PetscSqrtReal((PetscReal)test_size); /* 3-sigma rule */
      PetscReal std_tolerance  = 0.1 * std_target;                                       /* 10% tolerance for std dev */
      PetscReal skew_tolerance = 0.1;                                                    /* Skewness should be near 0 */
      PetscReal kurt_tolerance = 0.5;                                                    /* Excess kurtosis should be near 0 */

      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStatistical tests:\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean test     : %s (error %.6f < tolerance %.6f)\n", mean_error < mean_tolerance ? "PASS" : "FAIL", (double)mean_error, (double)mean_tolerance));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Std dev test  : %s (error %.6f < tolerance %.6f)\n", std_error < std_tolerance ? "PASS" : "FAIL", (double)std_error, (double)std_tolerance));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Skewness test : %s (|skewness| %.6f < tolerance %.6f)\n", PetscAbsReal(skewness) < skew_tolerance ? "PASS" : "FAIL", (double)PetscAbsReal(skewness), (double)skew_tolerance));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Kurtosis test : %s (|kurtosis| %.6f < tolerance %.6f)\n", PetscAbsReal(kurtosis) < kurt_tolerance ? "PASS" : "FAIL", (double)PetscAbsReal(kurtosis), (double)kurt_tolerance));

      /* Overall test result */
      PetscBool all_pass = (mean_error < mean_tolerance) && (std_error < std_tolerance) && (PetscAbsReal(skewness) < skew_tolerance) && (PetscAbsReal(kurtosis) < kurt_tolerance);
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nOverall result: %s\n", all_pass ? "PASS - Distribution is Gaussian" : "FAIL - Distribution may not be Gaussian"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==============================================\n\n"));

      PetscCall(VecDestroy(&test_vec));
    }
  }

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

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: 1
    args: -steps 112 -burn 10 -obs_freq 1 -obs_error 1 -petscda_view -petscda_ensemble_size 30

    test:
      requires: !complex
      suffix: eigen
      args: -petscda_ensemble_sqrt_type eigen

    test:
      suffix: chol
      args: -petscda_ensemble_sqrt_type cholesky

TEST*/
