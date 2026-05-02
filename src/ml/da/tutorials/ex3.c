static char help[] = "Shallow water test cases with data assimilation.\n"
                     "Implements 1D shallow water equations with 2 DOF per grid point (h, hu).\n\n"
                     "Example usage:\n"
                     "  ./ex3 -steps 100 -obs_freq 5 -obs_error 0.1 -petscda_view -petscda_ensemble_size 30\n"
                     "  ./ex3 -ex3_test wave -steps 500\n\n";

/* Data assimilation framework header (provides PetscDA) */
#include <petscda.h>
/* PETSc DMDA header (provides DM, DMDA functionality) */
#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscts.h>
#include <petscvec.h>

/* Default parameter values */
#define DEFAULT_N             80 /* 80 grid points */
#define DEFAULT_STEPS         100
#define DEFAULT_OBS_FREQ      5
#define DEFAULT_RANDOM_SEED   12345
#define DEFAULT_G             9.81
#define DEFAULT_DT            0.02
#define DEFAULT_OBS_ERROR_STD 0.01
#define DEFAULT_ENSEMBLE_SIZE 30
#define SPINUP_STEPS          0 /* No spinup needed - wave test has smooth analytical initial condition */

/* Minimum valid parameter values */
#define MIN_N                 1
#define MIN_ENSEMBLE_SIZE     2
#define MIN_OBS_FREQ          1
#define DEFAULT_PROGRESS_FREQ 10 /* Print progress every N steps by default */

/* Test case types */
typedef enum {
  EX3_TEST_DAM,
  EX3_TEST_WAVE
} Ex3TestType;

static PetscFunctionList Ex3TestList               = NULL;
static PetscBool         Ex3TestPackageInitialized = PETSC_FALSE;

typedef enum {
  EX3_FLUX_RUSANOV,
  EX3_FLUX_MC
} Ex3FluxType;

static const char *const Ex3FluxTypes[] = {"rusanov", "mc", "Ex3FluxType", "EX3_FLUX_", NULL};

typedef struct {
  DM          da;        /* 1D periodic DM storing the shallow water state */
  PetscInt    n_vert;    /* State dimension (number of grid points) */
  PetscReal   L;         /* Domain length */
  PetscReal   g;         /* Gravitational constant */
  PetscReal   dx;        /* Grid spacing */
  PetscReal   dt;        /* Integration time step size */
  TS          ts;        /* Reusable time stepper for efficiency */
  Ex3TestType test_type; /* Test case type */
  Ex3FluxType flux_type; /* Flux scheme */
} ShallowWaterCtx;

/*
  Limit - MC (Monotonized Central) limiter
*/
static PetscReal Limit(PetscReal a, PetscReal b)
{
  PetscReal c = 0.5 * (a + b);
  if (a * b <= 0.0) return 0.0;
  if (c > 0) return PetscMin(2.0 * a, PetscMin(2.0 * b, c));
  else return PetscMax(2.0 * a, PetscMax(2.0 * b, c));
}

/*
  ComputeFlux - Compute physical flux and wave speed for shallow water
*/
static void ComputeFlux(PetscReal g, PetscReal h, PetscReal hu, PetscReal *F_h, PetscReal *F_hu, PetscReal *u, PetscReal *c)
{
  if (h > 1e-10) {
    *u    = hu / h;
    *c    = PetscSqrtReal(g * h);
    *F_h  = hu;
    *F_hu = hu * *u + 0.5 * g * h * h;
  } else {
    *u    = 0.0;
    *c    = 0.0;
    *F_h  = 0.0;
    *F_hu = 0.0;
  }
}

/*
  ShallowWaterRHS - Compute the right-hand side of the shallow water equations

  Dispatches to appropriate flux scheme implementation.
*/
static PetscErrorCode ShallowWaterRHS(TS ts, PetscReal t, Vec X, Vec F_vec, PetscCtx ctx)
{
  ShallowWaterCtx   *sw = (ShallowWaterCtx *)ctx;
  Vec                X_local;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt           xs, xm, i;
  const PetscInt     ndof = 2; /* h and hu */

  PetscFunctionBeginUser;
  (void)ts;
  (void)t;

  PetscCall(DMDAGetCorners(sw->da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMGetLocalVector(sw->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayRead(sw->da, X_local, &x));
  PetscCall(DMDAVecGetArray(sw->da, F_vec, &f));

  if (sw->flux_type == EX3_FLUX_RUSANOV) {
    /* First-order Rusanov (Local Lax-Friedrichs) scheme */
    for (i = xs; i < xs + xm; i++) {
      PetscReal h  = PetscRealPart(x[i * ndof]);
      PetscReal hu = PetscRealPart(x[i * ndof + 1]);

      PetscReal h_im1  = PetscRealPart(x[(i - 1) * ndof]);
      PetscReal hu_im1 = PetscRealPart(x[(i - 1) * ndof + 1]);

      PetscReal h_ip1  = PetscRealPart(x[(i + 1) * ndof]);
      PetscReal hu_ip1 = PetscRealPart(x[(i + 1) * ndof + 1]);

      PetscReal F_h_i, F_hu_i, u, c;
      PetscReal F_h_im1, F_hu_im1, u_im1, c_im1;
      PetscReal F_h_ip1, F_hu_ip1, u_ip1, c_ip1;

      ComputeFlux(sw->g, h, hu, &F_h_i, &F_hu_i, &u, &c);
      ComputeFlux(sw->g, h_im1, hu_im1, &F_h_im1, &F_hu_im1, &u_im1, &c_im1);
      ComputeFlux(sw->g, h_ip1, hu_ip1, &F_h_ip1, &F_hu_ip1, &u_ip1, &c_ip1);

      PetscReal alpha_left  = PetscMax(PetscAbsReal(u_im1) + c_im1, PetscAbsReal(u) + c);
      PetscReal alpha_right = PetscMax(PetscAbsReal(u) + c, PetscAbsReal(u_ip1) + c_ip1);

      PetscReal flux_h_left  = 0.5 * (F_h_im1 + F_h_i - alpha_left * (h - h_im1));
      PetscReal flux_hu_left = 0.5 * (F_hu_im1 + F_hu_i - alpha_left * (hu - hu_im1));

      PetscReal flux_h_right  = 0.5 * (F_h_i + F_h_ip1 - alpha_right * (h_ip1 - h));
      PetscReal flux_hu_right = 0.5 * (F_hu_i + F_hu_ip1 - alpha_right * (hu_ip1 - hu));

      f[i * ndof]     = -(flux_h_right - flux_h_left) / sw->dx;
      f[i * ndof + 1] = -(flux_hu_right - flux_hu_left) / sw->dx;
    }
  } else {
    /* Second-order MC (Monotonized Central) scheme */
    for (i = xs; i < xs + xm; i++) {
      /* Read state */
      PetscReal h_im2 = PetscRealPart(x[(i - 2) * ndof]);
      PetscReal h_im1 = PetscRealPart(x[(i - 1) * ndof]);
      PetscReal h_i   = PetscRealPart(x[i * ndof]);
      PetscReal h_ip1 = PetscRealPart(x[(i + 1) * ndof]);
      PetscReal h_ip2 = PetscRealPart(x[(i + 2) * ndof]);

      PetscReal hu_im2 = PetscRealPart(x[(i - 2) * ndof + 1]);
      PetscReal hu_im1 = PetscRealPart(x[(i - 1) * ndof + 1]);
      PetscReal hu_i   = PetscRealPart(x[i * ndof + 1]);
      PetscReal hu_ip1 = PetscRealPart(x[(i + 1) * ndof + 1]);
      PetscReal hu_ip2 = PetscRealPart(x[(i + 2) * ndof + 1]);

      /* Compute slopes (MC limiter) */
      PetscReal s_h_im1 = Limit(h_im1 - h_im2, h_i - h_im1);
      PetscReal s_h_i   = Limit(h_i - h_im1, h_ip1 - h_i);
      PetscReal s_h_ip1 = Limit(h_ip1 - h_i, h_ip2 - h_ip1);

      PetscReal s_hu_im1 = Limit(hu_im1 - hu_im2, hu_i - hu_im1);
      PetscReal s_hu_i   = Limit(hu_i - hu_im1, hu_ip1 - hu_i);
      PetscReal s_hu_ip1 = Limit(hu_ip1 - hu_i, hu_ip2 - hu_ip1);

      /* Reconstruct states at interfaces */
      /* Left interface (i-1/2) */
      PetscReal h_L_left  = h_im1 + 0.5 * s_h_im1;
      PetscReal hu_L_left = hu_im1 + 0.5 * s_hu_im1;
      PetscReal h_R_left  = h_i - 0.5 * s_h_i;
      PetscReal hu_R_left = hu_i - 0.5 * s_hu_i;

      /* Right interface (i+1/2) */
      PetscReal h_L_right  = h_i + 0.5 * s_h_i;
      PetscReal hu_L_right = hu_i + 0.5 * s_hu_i;
      PetscReal h_R_right  = h_ip1 - 0.5 * s_h_ip1;
      PetscReal hu_R_right = hu_ip1 - 0.5 * s_hu_ip1;

      /* Compute fluxes */
      PetscReal F_h_LL, F_hu_LL, u_LL, c_LL;
      PetscReal F_h_RL, F_hu_RL, u_RL, c_RL;
      PetscReal F_h_LR, F_hu_LR, u_LR, c_LR;
      PetscReal F_h_RR, F_hu_RR, u_RR, c_RR;

      ComputeFlux(sw->g, h_L_left, hu_L_left, &F_h_LL, &F_hu_LL, &u_LL, &c_LL);
      ComputeFlux(sw->g, h_R_left, hu_R_left, &F_h_RL, &F_hu_RL, &u_RL, &c_RL);
      ComputeFlux(sw->g, h_L_right, hu_L_right, &F_h_LR, &F_hu_LR, &u_LR, &c_LR);
      ComputeFlux(sw->g, h_R_right, hu_R_right, &F_h_RR, &F_hu_RR, &u_RR, &c_RR);

      /* Rusanov flux */
      PetscReal speed_left   = PetscMax(PetscAbsReal(u_LL) + c_LL, PetscAbsReal(u_RL) + c_RL);
      PetscReal flux_h_left  = 0.5 * (F_h_LL + F_h_RL - speed_left * (h_R_left - h_L_left));
      PetscReal flux_hu_left = 0.5 * (F_hu_LL + F_hu_RL - speed_left * (hu_R_left - hu_L_left));

      PetscReal speed_right   = PetscMax(PetscAbsReal(u_LR) + c_LR, PetscAbsReal(u_RR) + c_RR);
      PetscReal flux_h_right  = 0.5 * (F_h_LR + F_h_RR - speed_right * (h_R_right - h_L_right));
      PetscReal flux_hu_right = 0.5 * (F_hu_LR + F_hu_RR - speed_right * (hu_R_right - hu_L_right));

      /* Update RHS using finite volume method */
      f[i * ndof]     = -(flux_h_right - flux_h_left) / sw->dx;
      f[i * ndof + 1] = -(flux_hu_right - flux_hu_left) / sw->dx;
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(sw->da, X_local, &x));
  PetscCall(DMDAVecRestoreArray(sw->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(sw->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterContextCreate - Create and initialize a shallow water context with reusable TS object
*/
static PetscErrorCode ShallowWaterContextCreate(DM da, PetscInt n_vert, PetscReal L, PetscReal g, PetscReal dt, Ex3TestType test_type, Ex3FluxType flux_type, ShallowWaterCtx **ctx)
{
  ShallowWaterCtx *sw;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sw));
  sw->da        = da;
  sw->n_vert    = n_vert;
  sw->L         = L;
  sw->g         = g;
  sw->dx        = L / n_vert; /* Domain is [0, L] */
  sw->dt        = dt;
  sw->test_type = test_type;
  sw->flux_type = flux_type;

  PetscCall(TSCreate(PetscObjectComm((PetscObject)da), &sw->ts));
  PetscCall(TSSetProblemType(sw->ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(sw->ts, NULL, ShallowWaterRHS, sw));
  PetscCall(TSSetType(sw->ts, TSRK));
  PetscCall(TSRKSetType(sw->ts, TSRK4));
  PetscCall(TSSetTimeStep(sw->ts, dt));
  PetscCall(TSSetMaxSteps(sw->ts, 1));
  PetscCall(TSSetMaxTime(sw->ts, dt));
  PetscCall(TSSetExactFinalTime(sw->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(sw->ts));
  /* Note: TSSetUp() will be called automatically by TSSolve() when needed */

  *ctx = sw;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterContextDestroy - Destroy a shallow water context and its TS object
*/
static PetscErrorCode ShallowWaterContextDestroy(ShallowWaterCtx **ctx)
{
  PetscFunctionBeginUser;
  if (!ctx || !*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterStep - Advance state vector one time step using shallow water dynamics
*/
static PetscErrorCode ShallowWaterStep(Vec input, Vec output, PetscCtx ctx)
{
  ShallowWaterCtx *sw = (ShallowWaterCtx *)ctx;

  PetscFunctionBeginUser;
  /* Copy input to output if they are different vectors */
  if (input != output) PetscCall(VecCopy(input, output));

  /* Reset the TS time for each integration (required for proper RK4 stepping) */
  PetscCall(TSSetTime(sw->ts, 0.0));
  PetscCall(TSSetStepNumber(sw->ts, 0));
  PetscCall(TSSetMaxTime(sw->ts, sw->dt));
  /* Solve one time step: advances output from t=0 to t=dt */
  PetscCall(TSSolve(sw->ts, output));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterSolution_Dam - Smooth periodic "dam-like" initial condition

  Creates a smooth Gaussian bump compatible with periodic boundaries.
  This avoids boundary artifacts while maintaining dam-like evolution.
*/
static PetscErrorCode ShallowWaterSolution_Dam(PetscReal L, PetscReal x, PetscReal *h, PetscReal *hu)
{
  const PetscReal h_mean = 1.5;      /* Mean water height */
  const PetscReal h_amp  = 0.4;      /* Bump amplitude */
  const PetscReal x_c    = 0.25 * L; /* Bump center */
  const PetscReal sigma  = 0.1 * L;  /* Gaussian width */

  PetscFunctionBeginUser;
  /* Smooth Gaussian bump: h = h_mean + h_amp * exp(-(x-x_c)^2/(2*sigma^2)) */
  PetscReal dx = x - x_c;
  /* Handle periodicity: use minimum distance on periodic domain */
  if (dx > 0.5 * L) dx -= L;
  if (dx < -0.5 * L) dx += L;

  *h = h_mean + h_amp * PetscExpReal(-dx * dx / (2.0 * sigma * sigma));
  /* Initially at rest */
  *hu = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterSolution_Wave - Traveling wave initial condition

  Sets smooth traveling wave with sinusoidal perturbation.
  For shallow water, a rightward-traveling wave requires velocity perturbation
  coupled to height: u' = c * (h'/h_mean) where c = sqrt(g*h_mean).
*/
static PetscErrorCode ShallowWaterSolution_Wave(PetscReal L, PetscReal x, PetscReal *h, PetscReal *hu)
{
  const PetscReal h_mean = 1.5;                       /* Mean water height */
  const PetscReal h_amp  = 0.3;                       /* Wave amplitude */
  const PetscReal g      = DEFAULT_G;                 /* Gravitational constant */
  const PetscReal k      = 2.0 * PETSC_PI / L;        /* Wave number (one wavelength over domain) */
  const PetscReal c      = PetscSqrtReal(g * h_mean); /* Wave speed */

  PetscFunctionBeginUser;
  /* Height field: h = h_mean + h_amp * sin(k*x) */
  PetscReal h_pert = h_amp * PetscSinReal(k * x);
  *h               = h_mean + h_pert;

  /* Velocity for rightward-traveling wave: u = c * (h'/h_mean)
     Using linearized shallow water: u ~= (c/h_mean) * h_pert */
  PetscReal u = (c / h_mean) * h_pert;
  *hu         = (*h) * u;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterSolution - Dispatch to appropriate initial condition based on test type
*/
static PetscErrorCode ShallowWaterSolution(Ex3TestType test_type, PetscReal L, PetscReal x, PetscReal *h, PetscReal *hu)
{
  PetscFunctionBeginUser;
  switch (test_type) {
  case EX3_TEST_DAM:
    PetscCall(ShallowWaterSolution_Dam(L, x, h, hu));
    break;
  case EX3_TEST_WAVE:
    PetscCall(ShallowWaterSolution_Wave(L, x, h, hu));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown test type");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateObservationMatrix - Create observation matrix H for shallow water, and H1 as scalar version

  Observes water height (h) at every other grid point.
  This creates a sparse matrix mapping from full state (n_vert*ndof) to observations.
  For n_vert=80 grid points, we observe at points 0, 2, 4, ..., 78
*/
static PetscErrorCode CreateObservationMatrix(PetscInt n_vert, PetscInt ndof, PetscInt nobs, Vec state, Mat *H, Mat *H1)
{
  PetscInt i, local_state_size;

  PetscFunctionBeginUser;
  PetscCheck(n_vert == 2 * nobs, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Number of grid points (%" PetscInt_FMT ") must equal 2*nobs (%" PetscInt_FMT ")", n_vert, 2 * nobs);

  PetscCall(VecGetLocalSize(state, &local_state_size));

  /* Create observation matrix H (nobs x n_vert*ndof) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, local_state_size, nobs, n_vert * ndof, 1, NULL, 0, NULL, H));
  PetscCall(MatSetFromOptions(*H));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, local_state_size / ndof, nobs, n_vert, 1, NULL, 0, NULL, H1));
  PetscCall(MatSetFromOptions(*H1));

  /* Observe water height (h) at every other grid point */
  for (i = 0; i < nobs; i++) {
    PetscInt grid_point = 2 * i; /* Observe at points 0, 2, 4, ... */
    PetscCall(MatSetValue(*H1, i, grid_point, 1.0, INSERT_VALUES));
    /* pick out the h component (first DOF) at that grid point */
    PetscCall(MatSetValue(*H, i, grid_point * ndof, 1.0, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(*H1, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H1, MAT_FINAL_ASSEMBLY));

  PetscCall(MatViewFromOptions(*H1, NULL, "-H_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateLocalizationMatrix - Create and initialize localization matrix Q for shallow water

  Q is a (num_vert x obs_size) matrix that specifies which observations affect each state variable.
  For no localization (global assimilation), each state variable uses all observations.
*/
static PetscErrorCode CreateLocalizationMatrix(PetscInt num_vert, PetscInt obs_size, Mat *Q)
{
  PetscInt i, j;

  PetscFunctionBeginUser;
  /* Create Q matrix (num_vert x obs_size)
     Each row will have obs_size non-zeros (all observations affect each state variable) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_vert, obs_size, obs_size, NULL, 0, NULL, Q));
  PetscCall(MatSetFromOptions(*Q));

  /* Initialize with no localization (global): each state variable uses all observations */
  for (i = 0; i < num_vert; i++) {
    for (j = 0; j < obs_size; j++) PetscCall(MatSetValue(*Q, i, j, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ValidateParameters - Validate input parameters and apply constraints
*/
static PetscErrorCode ValidateParameters(PetscInt *n, PetscInt *nobs, PetscInt *steps, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *g, PetscReal *obs_error_std)
{
  PetscFunctionBeginUser;
  PetscCheck(*n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "State dimension n must be positive, got %" PetscInt_FMT, *n);
  PetscCheck(*steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative, got %" PetscInt_FMT, *steps);
  PetscCheck(*ensemble_size >= MIN_ENSEMBLE_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least %d for meaningful statistics, got %" PetscInt_FMT, MIN_ENSEMBLE_SIZE, *ensemble_size);

  if (*obs_freq < MIN_OBS_FREQ) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency adjusted from %" PetscInt_FMT " to %d\n", *obs_freq, MIN_OBS_FREQ));
    *obs_freq = MIN_OBS_FREQ;
  }
  if (*obs_freq > *steps && *steps > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency (%" PetscInt_FMT ") > total steps (%" PetscInt_FMT "), no observations will be assimilated.\n", *obs_freq, *steps));

  PetscCheck(*dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step dt must be positive, got %g", (double)*dt);
  PetscCheck(*obs_error_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observation error std must be positive, got %g", (double)*obs_error_std);
  PetscCheck(PetscIsNormalReal(*g), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Gravitational constant g must be a normal real number");
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

/* Forward declaration */
static PetscErrorCode Ex3TestFinalizePackage(void);

/* Test type setters */
static PetscErrorCode Ex3SetTest_Dam(Ex3TestType *test_type)
{
  PetscFunctionBeginUser;
  *test_type = EX3_TEST_DAM;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Ex3SetTest_Wave(Ex3TestType *test_type)
{
  PetscFunctionBeginUser;
  *test_type = EX3_TEST_WAVE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Package initialization */
static PetscErrorCode Ex3TestInitializePackage(void)
{
  PetscFunctionBeginUser;
  if (Ex3TestPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  Ex3TestPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&Ex3TestList, "dam", Ex3SetTest_Dam));
  PetscCall(PetscFunctionListAdd(&Ex3TestList, "wave", Ex3SetTest_Wave));
  PetscCall(PetscRegisterFinalize(Ex3TestFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Ex3TestFinalizePackage(void)
{
  PetscFunctionBeginUser;
  Ex3TestPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&Ex3TestList));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  /* Configuration parameters */
  const PetscInt ndof                    = 2; /* Degrees of freedom per grid point: h and hu */
  PetscInt       n_vert                  = DEFAULT_N;
  PetscInt       steps                   = DEFAULT_STEPS;
  PetscInt       obs_freq                = DEFAULT_OBS_FREQ;
  PetscInt       random_seed             = DEFAULT_RANDOM_SEED;
  PetscInt       ensemble_size           = DEFAULT_ENSEMBLE_SIZE;
  PetscInt       n_spin                  = SPINUP_STEPS;
  PetscInt       progress_freq           = DEFAULT_PROGRESS_FREQ;
  PetscReal      g                       = DEFAULT_G;
  PetscReal      dt                      = DEFAULT_DT;
  PetscReal      obs_error_std           = DEFAULT_OBS_ERROR_STD;
  PetscBool      use_fake_localization   = PETSC_FALSE;
  PetscInt       num_observations_vertex = 7;
  PetscReal      L                       = (PetscReal)DEFAULT_N; /* Domain length */
  PetscReal      bd[3]                   = {L, 0, 0};
  Ex3TestType    test_type               = EX3_TEST_DAM;     /* Default to dam-break */
  Ex3FluxType    flux_type               = EX3_FLUX_RUSANOV; /* Default to first-order Rusanov */
  char           output_file[PETSC_MAX_PATH_LEN];
  PetscBool      output_enabled = PETSC_FALSE;
  FILE          *fp             = NULL;

  /* PETSc objects */
  ShallowWaterCtx *sw_ctx = NULL;
  DM               da_state;
  PetscDA          da;
  Vec              x0, x_mean, x_forecast;
  Vec              truth_state, rmse_work;
  Vec              observation, obs_noise, obs_error_var;
  PetscRandom      rng;
  Mat              Q = NULL;            /* Localization matrix */
  Mat              H = NULL, H1 = NULL; /* Observation operator matrix (h at every other grid point) and scalar version */

  /* Statistics tracking */
  PetscReal rmse_forecast = 0.0, rmse_analysis = 0.0;
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt  n_stat_steps = 0;
  PetscInt  obs_count    = 0;
  PetscInt  step;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Kokkos initialization deferred to Phase 5 optimization */

  /* Initialize test type package */
  PetscCall(Ex3TestInitializePackage());

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Shallow Water [L]ETKF Example", NULL);
  PetscCall(PetscOptionsInt("-n", "Number of grid points", "", n_vert, &n_vert, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsInt("-obs_freq", "Observation frequency", "", obs_freq, &obs_freq, NULL));
  PetscCall(PetscOptionsReal("-g", "Gravitational constant", "", g, &g, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation error standard deviation", "", obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsReal("-L", "Domain length", "", L, &L, NULL));
  bd[0] = L;
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble perturbations", "", random_seed, &random_seed, NULL));
  PetscCall(PetscOptionsInt("-progress_freq", "Print progress every N steps (0 = only first/last)", "", progress_freq, &progress_freq, NULL));
  PetscCall(PetscOptionsString("-output_file", "Output file for visualization data", "", "", output_file, sizeof(output_file), &output_enabled));
  PetscCall(PetscOptionsBool("-use_fake_localization", "Use fake localization matrix", "", use_fake_localization, &use_fake_localization, NULL));
  if (!use_fake_localization) PetscCall(PetscOptionsInt("-petscda_letkf_obs_per_vertex", "Number of observations per vertex", "", num_observations_vertex, &num_observations_vertex, NULL));
  else num_observations_vertex = n_vert;
  /* Parse test type option */
  {
    char        testTypeName[256];
    const char *defaultType                 = "dam";
    PetscBool   set                         = PETSC_FALSE;
    PetscErrorCode (*setter)(Ex3TestType *) = NULL;

    PetscCall(PetscStrncpy(testTypeName, defaultType, sizeof(testTypeName)));
    PetscCall(PetscOptionsFList("-ex3_test", "Test case type", "Ex3SetTest", Ex3TestList, defaultType, testTypeName, sizeof(testTypeName), &set));
    if (set) {
      PetscCall(PetscFunctionListFind(Ex3TestList, testTypeName, &setter));
      PetscCheck(setter, PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown test type \"%s\"", testTypeName);
      PetscCall((*setter)(&test_type));
    }
  }

  /* Parse flux type option */
  PetscCall(PetscOptionsEnum("-ex3_flux", "Flux scheme (rusanov/mc)", "", Ex3FluxTypes, (PetscEnum)flux_type, (PetscEnum *)&flux_type, NULL));
  n_spin = 0; /* No spinup needed for either test - dam evolves naturally, wave is already smooth */
  PetscOptionsEnd();

  /* LETKF constraint: nobs = n_vert/2, observe every other point */
  PetscInt nobs = n_vert / 2;

  /* Validate and constrain parameters */
  PetscCall(ValidateParameters(&n_vert, &nobs, &steps, &obs_freq, &ensemble_size, &dt, &g, &obs_error_std));

  /* Validate progress frequency */
  if (progress_freq < 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Progress frequency adjusted from %" PetscInt_FMT " to 0 (only first/last)\n", progress_freq));
    progress_freq = 0;
  }

  /* Create 1D periodic DM for state space with ndof=2 */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, n_vert, ndof, 2, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  /* Create shallow water context with reusable TS object */
  PetscCall(ShallowWaterContextCreate(da_state, n_vert, L, g, dt, test_type, flux_type, &sw_ctx));

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

  /* Set initial condition based on test type */
  {
    PetscScalar *x_array;
    PetscInt     xs, xm, i;
    PetscCall(DMDAGetCorners(da_state, &xs, NULL, NULL, &xm, NULL, NULL));
    PetscCall(DMDAVecGetArray(da_state, x0, &x_array));
    for (i = xs; i < xs + xm; i++) {
      PetscReal x = ((PetscReal)i + 0.5) * L / n_vert;
      PetscReal h, hu;
      PetscCall(ShallowWaterSolution(test_type, L, x, &h, &hu));
      x_array[i * ndof]     = h;
      x_array[i * ndof + 1] = hu;
    }
    PetscCall(DMDAVecRestoreArray(da_state, x0, &x_array));
  }

  /* Initialize truth trajectory */
  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecCopy(x0, truth_state));
  PetscCall(VecDuplicate(x0, &rmse_work));

  /* Spinup if needed (not used by default - both tests start from their analytical initial conditions) */
  if (n_spin > 0) {
    PetscInt spinup_progress_interval = (n_spin >= 10) ? (n_spin / 10) : 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinning up truth trajectory for %" PetscInt_FMT " steps...\n", n_spin));

    for (PetscInt k = 0; k < n_spin; k++) {
      PetscCall(ShallowWaterStep(truth_state, truth_state, sw_ctx));

      /* Progress reporting for long spinups */
      if ((k + 1) % spinup_progress_interval == 0 || (k + 1) == n_spin) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Spinup progress: %" PetscInt_FMT "/%" PetscInt_FMT " (%.0f%%)\n", k + 1, n_spin, 100.0 * (k + 1) / n_spin));
    }

    /* Update x0 to match spun-up state for consistent ensemble initialization */
    PetscCall(VecCopy(truth_state, x0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinup complete. Ensemble will be initialized from spun-up state.\n\n"));
  }

  /* Create observation matrix H, observing h at every other grid point) */
  PetscCall(CreateObservationMatrix(n_vert, ndof, nobs, x0, &H, &H1));

  /* Initialize observation vectors using MatCreateVecs from H (same as H1) */
  PetscCall(MatCreateVecs(H, NULL, &observation));
  PetscCall(VecDuplicate(observation, &obs_noise));
  PetscCall(VecDuplicate(observation, &obs_error_var));
  PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

  /* Create and configure PetscDA for ensemble data assimilation */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(PetscDASetSizes(da, n_vert * ndof, nobs));  /* State size includes ndof */
  PetscCall(PetscDAEnsembleSetSize(da, ensemble_size)); /* State size includes ndof */
  {
    PetscInt local_state_size, local_obs_size;
    PetscCall(VecGetLocalSize(x0, &local_state_size));
    PetscCall(VecGetLocalSize(observation, &local_obs_size));
    PetscCall(PetscDASetLocalSizes(da, local_state_size, local_obs_size));
  }
  PetscCall(PetscDASetNDOF(da, ndof)); /* Set number of degrees of freedom per grid point */
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
      Vec          Vecxyz[3] = {NULL, NULL, NULL};
      Vec          coord;
      DM           cda;
      PetscScalar *x_coord;
      PetscInt     xs, xm, i;

      /* Ensure coordinates are set */
      PetscCall(DMDASetUniformCoordinates(da_state, 0.0, L, 0.0, 0.0, 0.0, 0.0));
      /* Update coordinates to match cell centers as used in initial condition */
      PetscCall(DMGetCoordinateDM(da_state, &cda));
      PetscCall(DMGetCoordinates(da_state, &coord));
      PetscCall(DMDAGetCorners(cda, &xs, NULL, NULL, &xm, NULL, NULL));
      PetscCall(DMDAVecGetArray(cda, coord, &x_coord));
      for (i = xs; i < xs + xm; i++) x_coord[i] = ((PetscReal)i + 0.5) * L / n_vert;
      PetscCall(DMDAVecRestoreArray(cda, coord, &x_coord));

      /* Create Vecxyz[0] */
      PetscCall(DMCreateGlobalVector(cda, &Vecxyz[0]));
      PetscCall(VecSetFromOptions(Vecxyz[0]));
      PetscCall(PetscObjectSetName((PetscObject)Vecxyz[0], "x_coordinate"));
      PetscCall(VecCopy(coord, Vecxyz[0]));

      PetscCall(PetscDALETKFGetLocalizationMatrix(num_observations_vertex, 1, Vecxyz, bd, H1, &Q));
      PetscCall(VecDestroy(&Vecxyz[0]));
      PetscCall(PetscDALETKFSetObsPerVertex(da, num_observations_vertex));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization matrix Q created using PetscDALETKFGetLocalizationMatrix\n"));
    } else {
      PetscCall(CreateLocalizationMatrix(n_vert, nobs, &Q));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization matrix Q created: %dx%d, no localization/global (all weights = 1.0)\n", (int)n_vert, (int)nobs));
      if (isletkf) {
        PetscCall(PetscDALETKFSetObsPerVertex(da, num_observations_vertex)); // fully observed
      }
    }
    PetscCall(PetscDALETKFSetLocalization(da, Q, H));
    PetscCall(MatViewFromOptions(Q, NULL, "-Q_view"));
    PetscCall(MatDestroy(&Q));
  }

  /* Initialize ensemble members with perturbations around spun-up state
     This is critical for convergence - ensemble needs spread even after spinup */
  PetscCall(PetscDAEnsembleInitialize(da, x0, obs_error_std, rng));

  PetscCall(PetscDAViewFromOptions(da, NULL, "-petscda_view"));

  /* Print configuration summary */
  {
    const char *test_name = (test_type == EX3_TEST_DAM) ? "Dam-break" : "Traveling wave";
    const char *flux_name = (flux_type == EX3_FLUX_RUSANOV) ? "Rusanov (1st order)" : "MC (2nd order)";
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Shallow Water [L]ETKF Example\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "============================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "  Test case             : %s\n"
                          "  Flux scheme           : %s\n"
                          "  State dimension       : %" PetscInt_FMT " (%" PetscInt_FMT " grid points x %d DOF)\n"
                          "  Observation dimension : %" PetscInt_FMT "\n"
                          "  Ensemble size         : %" PetscInt_FMT "\n"
                          "  Domain length (L)     : %.4f\n"
                          "  Gravitational const   : %.4f\n"
                          "  Time step (dt)        : %.4f\n"
                          "  Total steps           : %" PetscInt_FMT "\n"
                          "  Observation frequency : %" PetscInt_FMT "\n"
                          "  Observation noise std : %.3f\n"
                          "  Random seed           : %" PetscInt_FMT "\n"
                          "  Localization          : None/Global (%" PetscInt_FMT " obs per vertex)\n\n",
                          test_name, flux_name, n_vert * ndof, n_vert, (int)ndof, nobs, ensemble_size, (double)L, (double)g, (double)dt, steps, obs_freq, (double)obs_error_std, random_seed, num_observations_vertex));
  }

  /* Open output file if requested */
  if (output_enabled) {
    PetscCall(PetscFOpen(PETSC_COMM_WORLD, output_file, "w", &fp));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# Shallow Water [L]ETKF Data Assimilation Output\n"));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# Test case: %s\n", (test_type == EX3_TEST_DAM) ? "Dam-break" : "Traveling wave"));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# n_vert=%d, ndof=%d, nobs=%d, ensemble_size=%d\n", (int)n_vert, (int)ndof, (int)nobs, (int)ensemble_size));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# dt=%.6f, g=%.6f, obs_error_std=%.6f\n", (double)dt, (double)g, (double)obs_error_std));
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "# Format: step time [truth_h truth_hu]x%d [mean_h mean_hu]x%d [obs]x%d\n", (int)n_vert, (int)n_vert, (int)nobs));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Writing output to: %s\n\n", output_file));

    /* Write initial condition (step 0) */
    const PetscScalar *truth_array, *mean_array;
    PetscInt           i;

    /* Compute initial ensemble mean */
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));

    PetscCall(DMDAVecGetArrayRead(da_state, truth_state, &truth_array));
    PetscCall(DMDAVecGetArrayRead(da_state, x_mean, &mean_array));

    /* Write step 0 and time 0 */
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "0 0.000000"));

    /* Write truth state (h, hu for each grid point) */
    for (i = 0; i < n_vert * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(truth_array[i])));

    /* Write ensemble mean (h, hu for each grid point) */
    for (i = 0; i < n_vert * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(mean_array[i])));

    /* Write nan for observations (no observations at step 0) */
    for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " nan"));

    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "\n"));

    PetscCall(DMDAVecRestoreArrayRead(da_state, truth_state, &truth_array));
    PetscCall(DMDAVecRestoreArrayRead(da_state, x_mean, &mean_array));
  }

  /* Print initial condition (step 0) */
  {
    PetscReal rmse_initial;
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
    PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, n_vert * ndof, &rmse_initial));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4d, time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f [initial]\n", 0, 0.0, (double)rmse_initial, (double)rmse_initial));
  }

  /* Main assimilation cycle: forecast and analysis steps */
  for (step = 1; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Propagate ensemble and truth trajectory from t_{k-1} to t_k */
    PetscCall(PetscDAEnsembleForecast(da, ShallowWaterStep, sw_ctx));
    PetscCall(ShallowWaterStep(truth_state, truth_state, sw_ctx));

    /* Forecast step: compute ensemble mean and forecast RMSE */
    PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, n_vert * ndof, &rmse_forecast));
    rmse_analysis = rmse_forecast;

    /* Analysis step: assimilate observations when available */
    if (step % obs_freq == 0 && step > 0) {
      /* Generate synthetic noisy observations from truth using observation matrix H */
      Vec truth_obs, temp_truth;
      PetscCall(MatCreateVecs(H, NULL, &truth_obs));
      PetscCall(MatCreateVecs(H, &temp_truth, NULL));

      /* Apply H to get observations: y = H*x_true
         Use temporary vector compatible with H's type to avoid Kokkos vector type issues */
      PetscCall(VecCopy(truth_state, temp_truth));
      PetscCall(MatMult(H, temp_truth, truth_obs));

      /* Add observation noise */
      PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
      PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_obs));

      /* Perform LETKF analysis with observation matrix H */
      PetscCall(PetscDAEnsembleAnalysis(da, observation, H));

      /* Clean up */
      PetscCall(VecDestroy(&temp_truth));
      PetscCall(VecDestroy(&truth_obs));

      /* Compute analysis RMSE */
      PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
      PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, n_vert * ndof, &rmse_analysis));
      obs_count++;
    }

    /* Accumulate statistics */
    sum_rmse_forecast += rmse_forecast;
    sum_rmse_analysis += rmse_analysis;
    n_stat_steps++;

    /* Write data to output file if enabled */
    if (output_enabled && fp) {
      const PetscScalar *truth_array, *mean_array, *obs_array;
      PetscInt           i;

      PetscCall(DMDAVecGetArrayRead(da_state, truth_state, &truth_array));
      PetscCall(DMDAVecGetArrayRead(da_state, x_mean, &mean_array));

      /* Write step and time */
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "%d %.6f", (int)step, (double)time));

      /* Write truth state (h, hu for each grid point) */
      for (i = 0; i < n_vert * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(truth_array[i])));

      /* Write ensemble mean (h, hu for each grid point) */
      for (i = 0; i < n_vert * ndof; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(mean_array[i])));

      /* Write observations (or nan if no observation at this step) */
      if (step % obs_freq == 0 && step > 0) {
        PetscCall(VecGetArrayRead(observation, &obs_array));
        for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " %.8e", (double)PetscRealPart(obs_array[i])));
        PetscCall(VecRestoreArrayRead(observation, &obs_array));
      } else {
        for (i = 0; i < nobs; i++) PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, " nan"));
      }

      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, fp, "\n"));

      PetscCall(DMDAVecRestoreArrayRead(da_state, truth_state, &truth_array));
      PetscCall(DMDAVecRestoreArrayRead(da_state, x_mean, &mean_array));
    }

    /* Progress reporting */
    if (progress_freq == 0) {
      /* Only print first and last steps */
      if (step == 0 || step == steps) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis));
    } else {
      /* Print every progress_freq steps, plus first and last */
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
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nWarning: No statistics collected\n\n"));
  }

  /* Close output file if opened */
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
  PetscCall(VecDestroy(&rmse_work));
  PetscCall(VecDestroy(&truth_state));
  PetscCall(VecDestroy(&x0));
  PetscCall(PetscDADestroy(&da));
  PetscCall(DMDestroy(&da_state));
  PetscCall(ShallowWaterContextDestroy(&sw_ctx));
  PetscCall(PetscRandomDestroy(&rng));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: kokkos_kernels !complex
    diff_args: -j
    args: -ex3_test dam -steps 10 -progress_freq 1 -petscda_view -petscda_ensemble_size 10 -obs_freq 2 -obs_error 0.03

    test:
      suffix: letkf_dam
      args: -petscda_type letkf -petscda_ensemble_size 7

    test:
      suffix: etkf_dam
      args: -petscda_ensemble_sqrt_type cholesky -petscda_type etkf

    test:
      nsize: 3
      suffix: kokkos_dam
      args: -petscda_type letkf -mat_type aijkokkos -vec_type kokkos -petscda_letkf_batch_size 13 -info :vec -petscda_ensemble_size 5 -petscda_letkf_obs_per_vertex 5

  testset:
    requires: kokkos_kernels !complex
    diff_args: -j
    args: -ex3_test wave -steps 10 -petscda_view -petscda_ensemble_size 10 -petscda_type letkf -obs_freq 2 -obs_error 0.03

    test:
      suffix: letkf_wave
      args: -petscda_type letkf -petscda_ensemble_size 5

    test:
      nsize: 3
      suffix: kokkos_wave
      args: -petscda_type letkf -mat_type aijkokkos -vec_type kokkos -petscda_letkf_batch_size 13 -info :vec -petscda_ensemble_size 5 -petscda_letkf_obs_per_vertex 5

    test:
      suffix: wave_mc
      args: -ex3_flux mc -petscda_type etkf

TEST*/
