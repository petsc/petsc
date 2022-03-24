static char help[] = "First example in homogenization book\n\n";

#include <petscsnes.h>
#include <petscdmplex.h>
#include <petscds.h>
#include <petscconvest.h>
#include <petscbag.h>

/*
  To control the refinement use -dm_plex_box_faces <n> or -dm_refine <k>, or both

  To see the exact and computed solutions

    -compare_view draw -draw_size 500,500 -draw_pause -1

  To see the delay in convergence of the discretization use

    -snes_convergence_estimate -convest_num_refine 7 -convest_monitor

  and to see the proper rate use

    -dm_refine 5 -snes_convergence_estimate -convest_num_refine 2 -convest_monitor
*/

typedef enum {MOD_CONSTANT, MOD_OSCILLATORY, MOD_TANH, NUM_MOD_TYPES} ModType;
const char *modTypes[NUM_MOD_TYPES+1] = {"constant", "oscillatory", "tanh", "unknown"};

/* Constants */
enum {EPSILON, NUM_CONSTANTS};

typedef struct {
  PetscReal epsilon; /* Wavelength of fine scale oscillation */
} Parameter;

typedef struct {
  PetscBag bag;      /* Holds problem parameters */
  ModType  modType;  /* Model type */
} AppCtx;

static PetscErrorCode trig_homogeneous_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 1.0;
  for (d = 0; d < dim; ++d) *u *= PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static PetscErrorCode oscillatory_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  Parameter      *param = (Parameter *) ctx;
  const PetscReal eps   = param->epsilon;

  u[0] = x[0] - x[0]*x[0] + (eps / (2.*PETSC_PI))*(0.5 - x[0])*PetscSinReal(2.*PETSC_PI*x[0]/eps) + PetscSqr(eps / (2.*PETSC_PI))*(1. - PetscCosReal(2.*PETSC_PI*x[0]/eps));
  return 0;
}

static void f0_trig_homogeneous_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    PetscScalar v = 1.;
    for (PetscInt e = 0; e < dim; e++) {
      if (e == d) {
        v *= -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
      } else {
        v *= PetscSinReal(2.0*PETSC_PI*x[d]);
      }
    }
    f0[0] += v;
  }
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static void f0_oscillatory_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -1.;
}

static void f1_oscillatory_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal eps = PetscRealPart(constants[EPSILON]);

  f1[0] = u_x[0] / (2. + PetscCosReal(2.*PETSC_PI*x[0]/eps));
}

static void g3_oscillatory_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal eps = PetscRealPart(constants[EPSILON]);

  g3[0] = 1. / (2. + PetscCosReal(2.*PETSC_PI*x[0]/eps));
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       mod;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->modType = MOD_CONSTANT;

  ierr = PetscOptionsBegin(comm, "", "Homogenization Problem Options", "DMPLEX");CHKERRQ(ierr);
  mod = options->modType;
  CHKERRQ(PetscOptionsEList("-mod_type", "The model type", "ex36.c", modTypes, NUM_MOD_TYPES, modTypes[options->modType], &mod, NULL));
  options->modType = (ModType) mod;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *user)
{
  PetscBag       bag;
  Parameter     *p;

  PetscFunctionBeginUser;
  CHKERRQ(PetscBagCreate(comm, sizeof(Parameter), &user->bag));
  CHKERRQ(PetscBagGetData(user->bag, (void **) &p));
  CHKERRQ(PetscBagSetName(user->bag, "par", "Homogenization parameters"));
  bag  = user->bag;
  CHKERRQ(PetscBagRegisterReal(bag, &p->epsilon, 1.0, "epsilon", "Wavelength of fine scale oscillation"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMSetApplicationContext(*dm, user));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS              ds;
  DMLabel              label;
  PetscSimplePointFunc ex;
  const PetscInt       id = 1;
  void                *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscBagGetData(user->bag, (void **) &ctx));
  switch (user->modType) {
    case MOD_CONSTANT:
      CHKERRQ(PetscDSSetResidual(ds, 0, f0_trig_homogeneous_u, f1_u));
      CHKERRQ(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
      CHKERRQ(DMGetLabel(dm, "marker", &label));
      ex   = trig_homogeneous_u;
      CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ex, NULL, ctx, NULL));
      break;
    case MOD_OSCILLATORY:
      CHKERRQ(PetscDSSetResidual(ds, 0, f0_oscillatory_u, f1_oscillatory_u));
      CHKERRQ(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_oscillatory_uu));
      CHKERRQ(DMGetLabel(dm, "marker", &label));
      ex   = oscillatory_u;
      CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ex, NULL, ctx, NULL));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_WRONG, "Unsupported model type: %s (%D)", modTypes[PetscMin(user->modType, NUM_MOD_TYPES)], user->modType);
  }
  CHKERRQ(PetscDSSetExactSolution(ds, 0, ex, ctx));
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[NUM_CONSTANTS];

    CHKERRQ(PetscBagGetData(user->bag, (void **) &param));

    constants[EPSILON] = param->epsilon;
    CHKERRQ(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscBool      simplex;
  PetscInt       dim;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  /* Create finite element */
  CHKERRQ(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ((*setup)(dm, user));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm,cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

static PetscErrorCode CompareView(Vec u)
{
  DM                dm;
  Vec               v[2], lv[2], exact;
  PetscOptions      options;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscInt          i;
  const char       *name, *prefix;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetDM(u, &dm));
  CHKERRQ(PetscObjectGetOptions((PetscObject) dm, &options));
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix));
  CHKERRQ(PetscOptionsGetViewer(PetscObjectComm((PetscObject) dm), options, prefix, "-compare_view", &viewer, &format, &flg));
  if (flg) {
    CHKERRQ(DMGetGlobalVector(dm, &exact));
    CHKERRQ(DMComputeExactSolution(dm, 0.0, exact, NULL));
    v[0] = u;
    v[1] = exact;
    for (i = 0; i < 2; ++i) {
      CHKERRQ(DMGetLocalVector(dm, &lv[i]));
      CHKERRQ(PetscObjectGetName((PetscObject) v[i], &name));
      CHKERRQ(PetscObjectSetName((PetscObject) lv[i], name));
      CHKERRQ(DMGlobalToLocalBegin(dm, v[i], INSERT_VALUES, lv[i]));
      CHKERRQ(DMGlobalToLocalEnd(dm, v[i], INSERT_VALUES, lv[i]));
      CHKERRQ(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, lv[i], 0., NULL, NULL, NULL));
    }
    CHKERRQ(DMPlexVecView1D(dm, 2, lv, viewer));
    for (i = 0; i < 2; ++i) CHKERRQ(DMRestoreLocalVector(dm, &lv[i]));
    CHKERRQ(DMRestoreGlobalVector(dm, &exact));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(0);
}

typedef struct
{
  Mat Mcoarse;   /* Mass matrix on the coarse space */
  Mat Mfine;     /* Mass matrix on the fine space */
  Mat Ifine;     /* Interpolator from coarse to fine */
  Vec Iscale;    /* Scaling vector for restriction */
  KSP kspCoarse; /* Solver for the coarse mass matrix */
  Vec tmpfine;   /* Temporary vector in the fine space */
  Vec tmpcoarse; /* Temporary vector in the coarse space */
} ProjStruct;

static PetscErrorCode DestroyCoarseProjection(Mat Pi)
{
  ProjStruct    *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Pi, (void **) &ctx));
  CHKERRQ(MatDestroy(&ctx->Mcoarse));
  CHKERRQ(MatDestroy(&ctx->Mfine));
  CHKERRQ(MatDestroy(&ctx->Ifine));
  CHKERRQ(VecDestroy(&ctx->Iscale));
  CHKERRQ(KSPDestroy(&ctx->kspCoarse));
  CHKERRQ(VecDestroy(&ctx->tmpcoarse));
  CHKERRQ(VecDestroy(&ctx->tmpfine));
  CHKERRQ(PetscFree(ctx));
  CHKERRQ(MatShellSetContext(Pi, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode CoarseProjection(Mat Pi, Vec x, Vec y)
{
  ProjStruct    *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Pi, (void **) &ctx));
  CHKERRQ(MatMult(ctx->Mfine, x, ctx->tmpfine));
  CHKERRQ(PetscObjectSetName((PetscObject) ctx->tmpfine, "Fine DG RHS"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpfine, "fine_dg_"));
  CHKERRQ(VecViewFromOptions(ctx->tmpfine, NULL, "-rhs_view"));
  CHKERRQ(MatMultTranspose(ctx->Ifine, ctx->tmpfine, ctx->tmpcoarse));
  CHKERRQ(PetscObjectSetName((PetscObject) ctx->tmpcoarse, "Coarse DG RHS"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpcoarse, "coarse_dg_"));
  CHKERRQ(VecViewFromOptions(ctx->tmpcoarse, NULL, "-rhs_view"));
  CHKERRQ(VecPointwiseMult(ctx->tmpcoarse, ctx->Iscale, ctx->tmpcoarse));
  CHKERRQ(VecViewFromOptions(ctx->tmpcoarse, NULL, "-rhs_view"));
  CHKERRQ(KSPSolve(ctx->kspCoarse, ctx->tmpcoarse, y));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateCoarseProjection(DM dmc, DM dmf, Mat *Pi)
{
  ProjStruct    *ctx;
  PetscInt       m, n, M, N;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(1, &ctx));
  CHKERRQ(DMCreateGlobalVector(dmc, &ctx->tmpcoarse));
  CHKERRQ(DMCreateGlobalVector(dmf, &ctx->tmpfine));
  CHKERRQ(VecGetLocalSize(ctx->tmpcoarse, &m));
  CHKERRQ(VecGetSize(ctx->tmpcoarse, &M));
  CHKERRQ(VecGetLocalSize(ctx->tmpfine, &n));
  CHKERRQ(VecGetSize(ctx->tmpfine, &N));
  CHKERRQ(DMCreateMassMatrix(dmc, dmc, &ctx->Mcoarse));
  CHKERRQ(DMCreateMassMatrix(dmf, dmf, &ctx->Mfine));
  CHKERRQ(DMCreateInterpolation(dmc, dmf, &ctx->Ifine, &ctx->Iscale));
  CHKERRQ(KSPCreate(PetscObjectComm((PetscObject) dmc), &ctx->kspCoarse));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) ctx->kspCoarse, "coarse_"));
  CHKERRQ(KSPSetOperators(ctx->kspCoarse, ctx->Mcoarse, ctx->Mcoarse));
  CHKERRQ(KSPSetFromOptions(ctx->kspCoarse));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject) dmc), m, n, M, N, ctx, Pi));
  CHKERRQ(MatShellSetOperation(*Pi, MATOP_DESTROY, (void (*)(void)) DestroyCoarseProjection));
  CHKERRQ(MatShellSetOperation(*Pi, MATOP_MULT, (void (*)(void)) CoarseProjection));
  PetscFunctionReturn(0);
}

typedef struct
{
  Mat Ifdg; /* Embed the fine space into the DG version */
  Mat Pi;   /* The L_2 stable projection to the DG coarse space */
  Vec tmpc; /* A temporary vector in the DG coarse space */
  Vec tmpf; /* A temporary vector in the DG fine space */
} QuasiInterp;

static PetscErrorCode DestroyQuasiInterpolator(Mat P)
{
  QuasiInterp   *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(P, (void **) &ctx));
  CHKERRQ(MatDestroy(&ctx->Ifdg));
  CHKERRQ(MatDestroy(&ctx->Pi));
  CHKERRQ(VecDestroy(&ctx->tmpc));
  CHKERRQ(VecDestroy(&ctx->tmpf));
  CHKERRQ(PetscFree(ctx));
  CHKERRQ(MatShellSetContext(P, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode QuasiInterpolate(Mat P, Vec x, Vec y)
{
  QuasiInterp   *ctx;
  DM             dmcdg, dmc;
  Vec            ly;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(P, (void **) &ctx));
  CHKERRQ(MatMult(ctx->Ifdg, x, ctx->tmpf));

  CHKERRQ(PetscObjectSetName((PetscObject) ctx->tmpf, "Fine DG Potential"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpf, "fine_dg_"));
  CHKERRQ(VecViewFromOptions(ctx->tmpf, NULL, "-vec_view"));
  CHKERRQ(MatMult(ctx->Pi, x, ctx->tmpc));

  CHKERRQ(PetscObjectSetName((PetscObject) ctx->tmpc, "Coarse DG Potential"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpc, "coarse_dg_"));
  CHKERRQ(VecViewFromOptions(ctx->tmpc, NULL, "-vec_view"));
  CHKERRQ(VecGetDM(ctx->tmpc, &dmcdg));

  CHKERRQ(VecGetDM(y, &dmc));
  CHKERRQ(DMGetLocalVector(dmc, &ly));
  CHKERRQ(DMPlexComputeClementInterpolant(dmcdg, ctx->tmpc, ly));
  CHKERRQ(DMLocalToGlobalBegin(dmc, ly, INSERT_VALUES, y));
  CHKERRQ(DMLocalToGlobalEnd(dmc, ly, INSERT_VALUES, y));
  CHKERRQ(DMRestoreLocalVector(dmc, &ly));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateQuasiInterpolator(DM dmc, DM dmf, Mat *P)
{
  QuasiInterp   *ctx;
  DM             dmcdg, dmfdg;
  PetscFE        fe;
  Vec            x, y;
  DMPolytopeType ct;
  PetscInt       dim, cStart, m, n, M, N;

  PetscFunctionBegin;
  CHKERRQ(PetscCalloc1(1, &ctx));
  CHKERRQ(DMGetGlobalVector(dmc, &x));
  CHKERRQ(DMGetGlobalVector(dmf, &y));
  CHKERRQ(VecGetLocalSize(x, &m));
  CHKERRQ(VecGetSize(x, &M));
  CHKERRQ(VecGetLocalSize(y, &n));
  CHKERRQ(VecGetSize(y, &N));
  CHKERRQ(DMRestoreGlobalVector(dmc, &x));
  CHKERRQ(DMRestoreGlobalVector(dmf, &y));

  CHKERRQ(DMClone(dmf, &dmfdg));
  CHKERRQ(DMGetDimension(dmfdg, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dmfdg, 0, &cStart, NULL));
  CHKERRQ(DMPlexGetCellType(dmfdg, cStart, &ct));
  CHKERRQ(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "fine_dg_", PETSC_DETERMINE, &fe));
  CHKERRQ(DMSetField(dmfdg, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dmfdg));
  CHKERRQ(DMCreateInterpolation(dmf, dmfdg, &ctx->Ifdg, NULL));
  CHKERRQ(DMCreateGlobalVector(dmfdg, &ctx->tmpf));
  CHKERRQ(DMDestroy(&dmfdg));

  CHKERRQ(DMClone(dmc, &dmcdg));
  CHKERRQ(DMGetDimension(dmcdg, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dmcdg, 0, &cStart, NULL));
  CHKERRQ(DMPlexGetCellType(dmcdg, cStart, &ct));
  CHKERRQ(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "coarse_dg_", PETSC_DETERMINE, &fe));
  CHKERRQ(DMSetField(dmcdg, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dmcdg));

  CHKERRQ(CreateCoarseProjection(dmcdg, dmf, &ctx->Pi));
  CHKERRQ(DMCreateGlobalVector(dmcdg, &ctx->tmpc));
  CHKERRQ(DMDestroy(&dmcdg));

  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject) dmc), m, n, M, N, ctx, P));
  CHKERRQ(MatShellSetOperation(*P, MATOP_DESTROY, (void (*)(void)) DestroyQuasiInterpolator));
  CHKERRQ(MatShellSetOperation(*P, MATOP_MULT, (void (*)(void)) QuasiInterpolate));
  PetscFunctionReturn(0);
}

static PetscErrorCode CoarseTest(DM dm, Vec u, AppCtx *user)
{
  DM             dmc;
  Mat            P;    /* The quasi-interpolator to the coarse space */
  Vec            uc;

  PetscFunctionBegin;
  if (user->modType == MOD_CONSTANT) PetscFunctionReturn(0);
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), &dmc));
  CHKERRQ(DMSetType(dmc, DMPLEX));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) dmc, "coarse_"));
  CHKERRQ(DMSetApplicationContext(dmc, user));
  CHKERRQ(DMSetFromOptions(dmc));
  CHKERRQ(DMViewFromOptions(dmc, NULL, "-dm_view"));

  CHKERRQ(SetupDiscretization(dmc, "potential", SetupPrimalProblem, user));
  CHKERRQ(DMCreateGlobalVector(dmc, &uc));
  CHKERRQ(PetscObjectSetName((PetscObject) uc, "potential"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) uc, "coarse_"));

  CHKERRQ(CreateQuasiInterpolator(dmc, dm, &P));
#if 1
  CHKERRQ(MatMult(P, u, uc));
#else
  {
    Mat In;
    Vec sc;

    CHKERRQ(DMCreateInterpolation(dmc, dm, &In, &sc));
    CHKERRQ(MatMultTranspose(In, u, uc));
    CHKERRQ(VecPointwiseMult(uc, sc, uc));
    CHKERRQ(MatDestroy(&In));
    CHKERRQ(VecDestroy(&sc));
  }
#endif
  CHKERRQ(CompareView(uc));

  CHKERRQ(MatDestroy(&P));
  CHKERRQ(VecDestroy(&uc));
  CHKERRQ(DMDestroy(&dmc));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(SetupParameters(PETSC_COMM_WORLD, &user));
  /* Primal system */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(SNESSetDM(snes, dm));
  CHKERRQ(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(VecSet(u, 0.0));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "potential"));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  CHKERRQ(SNESSetFromOptions(snes));
  CHKERRQ(DMSNESCheckFromOptions(snes, u));
  CHKERRQ(SNESSolve(snes, NULL, u));
  CHKERRQ(SNESGetSolution(snes, &u));
  CHKERRQ(VecViewFromOptions(u, NULL, "-potential_view"));
  CHKERRQ(CompareView(u));
  /* Looking at a coarse problem */
  CHKERRQ(CoarseTest(dm, u, &user));
  /* Cleanup */
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscBagDestroy(&user.bag));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1d_p1_constant
    args: -dm_plex_dim 1 -dm_plex_box_faces 4 -potential_petscspace_degree 1 -dmsnes_check

  test:
    suffix: 1d_p1_constant_conv
    args: -dm_plex_dim 1 -dm_plex_box_faces 4 -potential_petscspace_degree 1 \
          -snes_convergence_estimate -convest_num_refine 2

  test:
    suffix: 1d_p1_oscillatory
    args: -mod_type oscillatory -epsilon 0.03125 \
          -dm_plex_dim 1 -dm_plex_box_faces 4 -potential_petscspace_degree 1 -dm_refine 2 -dmsnes_check \
          -coarse_dm_plex_dim 1 -coarse_dm_plex_box_faces 4 -coarse_dm_plex_hash_location \
          -fine_dg_petscspace_degree 1 -fine_dg_petscdualspace_lagrange_continuity 0 \
          -coarse_dg_petscspace_degree 1 -coarse_dg_petscdualspace_lagrange_continuity 0

TEST*/
