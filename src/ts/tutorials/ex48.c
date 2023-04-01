static char help[] = "Magnetohydrodynamics (MHD) with Poisson brackets and "
                     "stream functions, solver testbed for M3D-C1. Used in https://arxiv.org/abs/2302.10242";

/*F
The strong form of a two field model for vorticity $\Omega$ and magnetic flux
$\psi$, using auxiliary variables potential $\phi$ and (negative) current
density $j_z$ \cite{Jardin04,Strauss98}.See http://arxiv.org/abs/  for more details
F*/

#include <assert.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

typedef enum _testidx {
  TEST_TILT,
  NUM_TEST_TYPES
} TestType;
const char *testTypes[NUM_TEST_TYPES + 1] = {"tilt", "unknown"};
typedef enum _modelidx {
  TWO_FILD,
  ONE_FILD,
  NUM_MODELS
} ModelType;
const char *modelTypes[NUM_MODELS + 1] = {"two-field", "one-field", "unknown"};
typedef enum _fieldidx {
  JZ,
  PSI,
  PHI,
  OMEGA,
  NUM_COMP
} FieldIdx; // add more
typedef enum _const_idx {
  MU_CONST,
  ETA_CONST,
  TEST_CONST,
  NUM_CONSTS
} ConstIdx;

typedef struct {
  PetscInt  debug; /* The debugging level */
  PetscReal plotDt;
  PetscReal plotStartTime;
  PetscInt  plotIdx;
  PetscInt  plotStep;
  PetscBool plotting;
  PetscInt  dim;                          /* The topological mesh dimension */
  char      filename[PETSC_MAX_PATH_LEN]; /* The optional ExodusII file */
  PetscErrorCode (**initialFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscReal mu, eta;
  PetscReal perturb;
  TestType  testType;
  ModelType modelType;
  PetscInt  Nf;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt ii;

  PetscFunctionBeginUser;
  options->debug         = 1;
  options->filename[0]   = '\0';
  options->testType      = TEST_TILT;
  options->modelType     = TWO_FILD;
  options->mu            = 0.005;
  options->eta           = 0.001;
  options->perturb       = 0;
  options->plotDt        = 0.1;
  options->plotStartTime = 0.0;
  options->plotIdx       = 0;
  options->plotStep      = PETSC_MAX_INT;
  options->plotting      = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "MHD Problem Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-debug", "The debugging level", "mhd.c", options->debug, &options->debug, NULL));
  ii                = (PetscInt)options->testType;
  options->testType = TEST_TILT;
  ii                = options->testType;
  PetscCall(PetscOptionsEList("-test_type", "The test type: 'tilt' Tilt instability", "mhd.c", testTypes, NUM_TEST_TYPES, testTypes[options->testType], &ii, NULL));
  options->testType  = (TestType)ii;
  ii                 = (PetscInt)options->modelType;
  options->modelType = TWO_FILD;
  ii                 = options->modelType;
  PetscCall(PetscOptionsEList("-model_type", "The model type: 'two', 'one' field", "mhd.c", modelTypes, NUM_MODELS, modelTypes[options->modelType], &ii, NULL));
  options->modelType = (ModelType)ii;
  options->Nf        = options->modelType == TWO_FILD ? 4 : 2;

  PetscCall(PetscOptionsReal("-mu", "Magnetic resistivity", "mhd.c", options->mu, &options->mu, NULL));
  PetscCall(PetscOptionsReal("-eta", "Viscosity", "mhd.c", options->eta, &options->eta, NULL));
  PetscCall(PetscOptionsReal("-plot_dt", "Plot frequency in time", "mhd.c", options->plotDt, &options->plotDt, NULL));
  PetscCall(PetscOptionsReal("-plot_start_time", "Time to delay start of plotting", "mhd.c", options->plotStartTime, &options->plotStartTime, NULL));
  PetscCall(PetscOptionsReal("-perturbation", "Random perturbation of initial psi scale", "mhd.c", options->perturb, &options->perturb, NULL));
  PetscCall(PetscPrintf(comm, "Test Type = %s\n", testTypes[options->testType]));
  PetscCall(PetscPrintf(comm, "Model Type = %s\n", modelTypes[options->modelType]));
  PetscCall(PetscPrintf(comm, "eta = %g\n", (double)options->eta));
  PetscCall(PetscPrintf(comm, "mu = %g\n", (double)options->mu));
  PetscOptionsEnd();

  PetscFunctionReturn(PETSC_SUCCESS);
}

// | 0 1 | matrix to apply bracket
// |-1 0 |
static PetscReal s_K[2][2] = {
  {0,  1},
  {-1, 0}
};

/*
 dt - "g0" are mass terms
*/
static void g0_dt(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift;
}

/*
 Identity, Mass
*/
static void g0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1;
}
/* 'right' Poisson bracket -<.,phi>, linearized variable is left (column), data
 * variable right */
static void g1_phi_right(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt           i, j;
  const PetscScalar *pphiDer = &u_x[uOff_x[PHI]]; // get derivative of the 'right' ("dg") and apply to
                                                  // live var "df"
  for (i = 0; i < dim; ++i)
    for (j = 0; j < dim; ++j)
      //  indexing with inner, j, index generates the left live variable [dy,-]
      //  by convention, put j index on right, with i destination: [ d/dy,
      //  -d/dx]'
      g1[i] += s_K[i][j] * pphiDer[j];
}
/* 'left' bracket -{jz,.}, "n" for negative, linearized variable right (column),
 * data variable left */
static void g1_njz_left(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt           i, j;
  const PetscScalar *jzDer = &u_x[uOff_x[JZ]]; // get derivative of the 'left' ("df") and apply to live
                                               // var "dg"
  for (i = 0; i < dim; ++i)
    for (j = 0; j < dim; ++j)
      // live right: Der[i] * K: Der[j] --> j: [d/dy, -d/dx]'
      g1[j] += -jzDer[i] * s_K[i][j];
}
/* 'left' Poisson bracket -< . , psi> */
static void g1_npsi_right(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt           i, j;
  const PetscScalar *psiDer = &u_x[uOff_x[PSI]];
  for (i = 0; i < dim; ++i)
    for (j = 0; j < dim; ++j) g1[i] += -s_K[i][j] * psiDer[j];
}

/* < Omega , . > */
static void g1_omega_left(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt           i, j;
  const PetscScalar *pOmegaDer = &u_x[uOff_x[OMEGA]];
  for (i = 0; i < dim; ++i)
    for (j = 0; j < dim; ++j) g1[j] += pOmegaDer[i] * s_K[i][j];
}

/* < psi , . > */
static void g1_psi_left(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt           i, j;
  const PetscScalar *pPsiDer = &u_x[uOff_x[PSI]];
  for (i = 0; i < dim; ++i)
    for (j = 0; j < dim; ++j) g1[j] += pPsiDer[i] * s_K[i][j];
}

// -Lapacians (resistivity), negative sign goes away from IBP
static void g3_nmu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal mu = PetscRealPart(constants[MU_CONST]);
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = mu;
}

// Auxiliary variable = -del^2 x, negative sign goes away from IBP
static void g3_n1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1;
}

/* residual point methods */
static PetscScalar poissonBracket(PetscInt dim, const PetscScalar df[], const PetscScalar dg[])
{
  PetscScalar ret = df[0] * dg[1] - df[1] * dg[0];
  if (dim == 3) {
    ret += df[1] * dg[2] - df[2] * dg[1];
    ret += df[2] * dg[0] - df[0] * dg[2];
  }
  return ret;
}
//
static void f0_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar *omegaDer = &u_x[uOff_x[OMEGA]];
  const PetscScalar *psiDer   = &u_x[uOff_x[PSI]];
  const PetscScalar *phiDer   = &u_x[uOff_x[PHI]];
  const PetscScalar *jzDer    = &u_x[uOff_x[JZ]];

  f0[0] += poissonBracket(dim, omegaDer, phiDer) - poissonBracket(dim, jzDer, psiDer);

  if (u_t) f0[0] += u_t[OMEGA];
}

static void f1_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *omegaDer = &u_x[uOff_x[OMEGA]];
  PetscReal          mu       = PetscRealPart(constants[MU_CONST]);

  for (PetscInt d = 0; d < dim; ++d) f1[d] += mu * omegaDer[d];
}

// d/dt + {psi,phi} - eta j_z
static void f0_psi_4f(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar *psiDer = &u_x[uOff_x[PSI]];
  const PetscScalar *phiDer = &u_x[uOff_x[PHI]];
  PetscReal          eta    = PetscRealPart(constants[ETA_CONST]);

  f0[0] = -eta * u[uOff[JZ]];
  f0[0] += poissonBracket(dim, psiDer, phiDer);

  if (u_t) f0[0] += u_t[PSI];
  // printf("psiDer = %20.15e %20.15e psi = %20.15e
}

// d/dt - eta j_z
static void f0_psi_2f(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal eta = PetscRealPart(constants[ETA_CONST]);

  f0[0] = -eta * u[uOff[JZ]];

  if (u_t) f0[0] += u_t[PSI];
  // printf("psiDer = %20.15e %20.15e psi = %20.15e
}

static void f0_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += u[uOff[OMEGA]];
}

static void f1_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *phiDer = &u_x[uOff_x[PHI]];

  for (PetscInt d = 0; d < dim; ++d) f1[d] = phiDer[d];
}

/* - eta M */
static void g0_neta(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscReal eta = PetscRealPart(constants[ETA_CONST]);

  g0[0] = -eta;
}

static void f0_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[uOff[JZ]];
}

/* -del^2 psi = (grad v, grad psi) */
static void f1_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscScalar *psiDer = &u_x[uOff_x[PSI]];

  for (PetscInt d = 0; d < dim; ++d) f1[d] = psiDer[d];
}

static void f0_mhd_B_energy2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  const PetscScalar *psiDer = &u_x[uOff_x[PSI]];
  PetscScalar        b2     = 0;
  for (int i = 0; i < dim; ++i) b2 += psiDer[i] * psiDer[i];
  f0[0] = b2;
}

static void f0_mhd_v_energy2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  const PetscScalar *phiDer = &u_x[uOff_x[PHI]];
  PetscScalar        v2     = 0;
  for (int i = 0; i < dim; ++i) v2 += phiDer[i] * phiDer[i];
  f0[0] = v2;
}

static PetscErrorCode Monitor(TS ts, PetscInt stepi, PetscReal time, Vec X, void *actx)
{
  AppCtx             *ctx = (AppCtx *)actx; /* user-defined application context */
  SNES                snes;
  SNESConvergedReason reason;
  TSConvergedReason   tsreason;

  PetscFunctionBegin;
  // PetscCall(TSGetApplicationContext(ts, &ctx));
  if (ctx->debug < 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  if (reason < 0) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "\t\t ***************** Monitor: SNES diverged with reason %d.\n", (int)reason));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (stepi > ctx->plotStep && ctx->plotting) {
    ctx->plotting = PETSC_FALSE; /* was doing diagnostics, now done */
    ctx->plotIdx++;
  }
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetConvergedReason(ts, &tsreason));
  if (((time - ctx->plotStartTime) / ctx->plotDt >= (PetscReal)ctx->plotIdx && time >= ctx->plotStartTime) || (tsreason == TS_CONVERGED_TIME || tsreason == TS_CONVERGED_ITS) || ctx->plotIdx == 0) {
    DM          dm, plex;
    Vec         X;
    PetscReal   val;
    PetscScalar tt[12]; // FE integral seems to need a large array
    PetscDS     prob;
    if (!ctx->plotting) { /* first step of possible backtracks */
      ctx->plotting = PETSC_TRUE;
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\t\t ?????? ------\n"));
      ctx->plotting = PETSC_TRUE;
    }
    ctx->plotStep = stepi;
    PetscCall(TSGetSolution(ts, &X));
    PetscCall(VecGetDM(X, &dm));
    PetscCall(DMGetOutputSequenceNumber(dm, NULL, &val));
    PetscCall(DMSetOutputSequenceNumber(dm, ctx->plotIdx, val));
    PetscCall(VecViewFromOptions(X, NULL, "-vec_view_mhd"));
    if (ctx->debug > 2) {
      Vec R;
      PetscCall(SNESGetFunction(snes, &R, NULL, NULL));
      PetscCall(VecViewFromOptions(R, NULL, "-vec_view_res"));
    }
    // compute energy
    PetscCall(DMGetDS(dm, &prob));
    PetscCall(DMConvert(dm, DMPLEX, &plex));
    PetscCall(PetscDSSetObjective(prob, 0, &f0_mhd_v_energy2));
    PetscCall(DMPlexComputeIntegralFEM(plex, X, &tt[0], ctx));
    val = PetscRealPart(tt[0]);
    PetscCall(PetscDSSetObjective(prob, 0, &f0_mhd_B_energy2));
    PetscCall(DMPlexComputeIntegralFEM(plex, X, &tt[0], ctx));
    val = PetscSqrtReal(val) * 0.5 + PetscSqrtReal(PetscRealPart(tt[0])) * 0.5;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "MHD %4d) time = %9.5g, Eergy= %20.13e (plot ID %d)\n", (int)ctx->plotIdx, (double)time, (double)val, (int)ctx->plotIdx));
    /* clean up */
    PetscCall(DMDestroy(&plex));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DMLabel label;
  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, name));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, PETSC_DETERMINE, label));
  PetscCall(DMPlexLabelComplete(dm, label));
  PetscFunctionReturn(PETSC_SUCCESS);
}
// Create mesh, dim is set here
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  const char *filename = ctx->filename;
  size_t      len;
  char        buff[256];
  PetscMPIInt size;
  PetscInt    nface = 1;
  PetscFunctionBeginUser;
  PetscCall(PetscStrlen(filename, &len));
  if (len) {
    PetscCall(DMPlexCreateFromFile(comm, filename, "", PETSC_TRUE, dm));
  } else {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
  }
  PetscCallMPI(MPI_Comm_size(comm, &size));
  while (nface * nface < size) nface *= 2; // 2D
  if (nface < 2) nface = 2;
  PetscCall(PetscSNPrintf(buff, sizeof(buff), "-dm_plex_box_faces %d,%d -petscpartitioner_type simple", (int)nface, (int)nface));
  PetscCall(PetscOptionsInsertString(NULL, buff));
  PetscCall(PetscOptionsInsertString(NULL, "-dm_plex_box_lower -2,-2 -dm_plex_box_upper 2,2"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  PetscCall(DMGetDimension(*dm, &ctx->dim));
  {
    char      convType[256];
    PetscBool flg;
    PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");
    PetscCall(PetscOptionsFList("-dm_plex_convert_type", "Convert DMPlex to another format", "mhd", DMList, DMPLEX, convType, 256, &flg));
    PetscOptionsEnd();
    if (flg) {
      DM dmConv;
      PetscCall(DMConvert(*dm, convType, &dmConv));
      if (dmConv) {
        PetscCall(DMDestroy(dm));
        *dm = dmConv;
      }
    }
  }
  PetscCall(DMLocalizeCoordinates(*dm)); /* needed for periodic */
  {
    PetscBool hasLabel;
    PetscCall(DMHasLabel(*dm, "marker", &hasLabel));
    if (!hasLabel) PetscCall(CreateBCLabel(*dm, "marker"));
  }
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Mesh"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view_mhd"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view_res"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode initialSolution_Omega(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode initialSolution_Psi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *a_ctx)
{
  AppCtx   *ctx = (AppCtx *)a_ctx;
  PetscReal r   = 0, theta, cos_theta;
  // k = sp.jn_zeros(1, 1)[0]
  const PetscReal k = 3.8317059702075125;
  for (PetscInt i = 0; i < dim; i++) r += x[i] * x[i];
  r = PetscSqrtReal(r);
  // r = sqrt(dot(x,x))
  theta     = PetscAtan2Real(x[1], x[0]);
  cos_theta = PetscCosReal(theta);
  // f = conditional(gt(r, 1.0), outer_f, inner_f)
  if (r < 1.0) {
    // inner_f =
    // (2/(Constant(k)*bessel_J(0,Constant(k))))*bessel_J(1,Constant(k)*r)*cos_theta
    u[0] = 2.0 / (k * j0(k)) * j1(k * r) * cos_theta;
  } else {
    // outer_f =  (1/r - r)*cos_theta
    u[0] = (r - 1.0 / r) * cos_theta;
  }
  u[0] += ctx->perturb * ((double)rand() / (double)RAND_MAX - 0.5);
  return PETSC_SUCCESS;
}

static PetscErrorCode initialSolution_Phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode initialSolution_Jz(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode SetupProblem(PetscDS prob, DM dm, AppCtx *ctx)
{
  PetscInt f;

  PetscFunctionBeginUser;
  // for both 2 & 4 field (j_z is same)
  PetscCall(PetscDSSetJacobian(prob, JZ, JZ, g0_1, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(prob, JZ, PSI, NULL, NULL, NULL, g3_n1));
  PetscCall(PetscDSSetResidual(prob, JZ, f0_jz, f1_jz));

  PetscCall(PetscDSSetJacobian(prob, PSI, JZ, g0_neta, NULL, NULL, NULL));
  if (ctx->modelType == ONE_FILD) {
    PetscCall(PetscDSSetJacobian(prob, PSI, PSI, g0_dt, NULL, NULL,
                                 NULL)); // remove phi term

    PetscCall(PetscDSSetResidual(prob, PSI, f0_psi_2f, NULL));
  } else {
    PetscCall(PetscDSSetJacobian(prob, PSI, PSI, g0_dt, g1_phi_right, NULL, NULL));
    PetscCall(PetscDSSetJacobian(prob, PSI, PHI, NULL, g1_psi_left, NULL, NULL));
    PetscCall(PetscDSSetResidual(prob, PSI, f0_psi_4f, NULL));

    PetscCall(PetscDSSetJacobian(prob, PHI, PHI, NULL, NULL, NULL, g3_n1));
    PetscCall(PetscDSSetJacobian(prob, PHI, OMEGA, g0_1, NULL, NULL, NULL));
    PetscCall(PetscDSSetResidual(prob, PHI, f0_phi, f1_phi));

    PetscCall(PetscDSSetJacobian(prob, OMEGA, OMEGA, g0_dt, g1_phi_right, NULL, g3_nmu));
    PetscCall(PetscDSSetJacobian(prob, OMEGA, PSI, NULL, g1_njz_left, NULL, NULL));
    PetscCall(PetscDSSetJacobian(prob, OMEGA, PHI, NULL, g1_omega_left, NULL, NULL));
    PetscCall(PetscDSSetJacobian(prob, OMEGA, JZ, NULL, g1_npsi_right, NULL, NULL));
    PetscCall(PetscDSSetResidual(prob, OMEGA, f0_Omega, f1_Omega));
  }
  /* Setup constants - is this persistent? */
  {
    PetscScalar scales[NUM_CONSTS]; // +1 adding in testType for use in the f
                                    // and g functions
    /* These could be set from the command line */
    scales[MU_CONST]  = ctx->mu;
    scales[ETA_CONST] = ctx->eta;
    // scales[TEST_CONST] = (PetscReal)ctx->testType; -- how to make work with complex
    PetscCall(PetscDSSetConstants(prob, NUM_CONSTS, scales));
  }
  for (f = 0; f < ctx->Nf; ++f) {
    ctx->initialFuncs[f] = NULL;
    PetscCall(PetscDSSetImplicit(prob, f, PETSC_TRUE));
  }
  if (ctx->testType == TEST_TILT) {
    const PetscInt id = 1;
    DMLabel        label;
    PetscCall(DMGetLabel(dm, "marker", &label));

    ctx->initialFuncs[JZ]  = initialSolution_Jz;
    ctx->initialFuncs[PSI] = initialSolution_Psi;

    PetscCall(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Jz for tilt test", label, 1, &id, JZ, 0, NULL, (void (*)(void))initialSolution_Jz, NULL, ctx, NULL));
    PetscCall(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Psi for tilt test", label, 1, &id, PSI, 0, NULL, (void (*)(void))initialSolution_Psi, NULL, ctx, NULL));
    if (ctx->modelType == TWO_FILD) {
      ctx->initialFuncs[OMEGA] = initialSolution_Omega;
      ctx->initialFuncs[PHI]   = initialSolution_Phi;
      PetscCall(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Omega for tilt test", label, 1, &id, OMEGA, 0, NULL, (void (*)(void))initialSolution_Omega, NULL, ctx, NULL));
      PetscCall(PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Phi for tilt test", label, 1, &id, PHI, 0, NULL, (void (*)(void))initialSolution_Phi, NULL, ctx, NULL));
    }
  } else {
    PetscCheck(0, PetscObjectComm((PetscObject)prob), PETSC_ERR_ARG_WRONG, "Unsupported test type: %s (%d)", testTypes[PetscMin(ctx->testType, NUM_TEST_TYPES)], (int)ctx->testType);
  }
  PetscCall(PetscDSSetContext(prob, 0, ctx));
  PetscCall(PetscDSSetFromOptions(prob));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  DM             cdm;
  const PetscInt dim = ctx->dim;
  PetscFE        fe[NUM_COMP];
  PetscDS        prob;
  PetscInt       Nf           = ctx->Nf, f;
  PetscBool      cell_simplex = PETSC_TRUE;
  MPI_Comm       comm         = PetscObjectComm((PetscObject)dm);

  PetscFunctionBeginUser;
  /* Create finite element */
  PetscCall(PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[JZ]));
  PetscCall(PetscObjectSetName((PetscObject)fe[JZ], "j_z"));
  PetscCall(DMSetField(dm, JZ, NULL, (PetscObject)fe[JZ]));
  PetscCall(PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[PSI]));
  PetscCall(PetscObjectSetName((PetscObject)fe[PSI], "psi"));
  PetscCall(DMSetField(dm, PSI, NULL, (PetscObject)fe[PSI]));
  if (ctx->modelType == TWO_FILD) {
    PetscCall(PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[OMEGA]));
    PetscCall(PetscObjectSetName((PetscObject)fe[OMEGA], "Omega"));
    PetscCall(DMSetField(dm, OMEGA, NULL, (PetscObject)fe[OMEGA]));

    PetscCall(PetscFECreateDefault(comm, dim, 1, cell_simplex, NULL, -1, &fe[PHI]));
    PetscCall(PetscObjectSetName((PetscObject)fe[PHI], "phi"));
    PetscCall(DMSetField(dm, PHI, NULL, (PetscObject)fe[PHI]));
  }
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &prob));
  for (f = 0; f < Nf; ++f) PetscCall(PetscDSSetDiscretization(prob, f, (PetscObject)fe[f]));
  PetscCall(SetupProblem(prob, dm, ctx));
  cdm = dm;
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    if (dm != cdm) PetscCall(PetscObjectSetName((PetscObject)cdm, "Coarse"));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  for (f = 0; f < Nf; ++f) PetscCall(PetscFEDestroy(&fe[f]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  TS          ts;
  Vec         u, r;
  AppCtx      ctx;
  PetscReal   t        = 0.0;
  AppCtx     *ctxarr[] = {&ctx, &ctx, &ctx, &ctx}; // each variable could have a different context
  PetscMPIInt rank;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx)); // dim is not set
  /* create mesh and problem */
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMSetApplicationContext(dm, &ctx));
  PetscCall(PetscMalloc1(ctx.Nf, &ctx.initialFuncs));
  PetscCall(SetupDiscretization(dm, &ctx));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(VecDuplicate(u, &r));
  PetscCall(PetscObjectSetName((PetscObject)r, "r"));
  /* create TS */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(TSSetApplicationContext(ts, &ctx));
  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx));
  PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx));
  PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetMaxTime(ts, 15.0));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSMonitorSet(ts, Monitor, &ctx, NULL));
  /* make solution */
  PetscCall(DMProjectFunction(dm, t, ctx.initialFuncs, (void **)ctxarr, INSERT_ALL_VALUES, u));
  ctx.perturb = 0.0;
  PetscCall(TSSetSolution(ts, u));
  // solve
  PetscCall(TSSolve(ts, u));
  // cleanup
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFree(ctx.initialFuncs));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: triangle !complex
    nsize: 4
    args: -dm_plex_box_lower -2,-2 -dm_plex_box_upper 2,2 -dm_plex_simplex 1 -dm_refine_hierarchy 2 \
      -eta 0.0001 -ksp_converged_reason -ksp_max_it 50 -ksp_rtol 1e-3 -ksp_type fgmres -mg_coarse_ksp_rtol 1e-1 \
      -mg_coarse_ksp_type fgmres -mg_coarse_mg_levels_ksp_type gmres -mg_coarse_pc_type gamg -mg_levels_ksp_max_it 4 \
      -mg_levels_ksp_type gmres -mg_levels_pc_type jacobi -mu 0.005 -pc_mg_type full -pc_type mg \
      -petscpartitioner_type simple -petscspace_degree 2 -snes_converged_reason -snes_max_it 10 -snes_monitor \
      -snes_rtol 1.e-9 -snes_stol 1.e-9 -ts_adapt_dt_max 0.01 -ts_adapt_monitor -ts_arkimex_type 1bee \
      -ts_dt 0.001 -ts_max_reject 10 -ts_max_snes_failures -1 -ts_max_steps 1 -ts_max_time -ts_monitor -ts_type arkimex
    filter: grep -v DM_

TEST*/
