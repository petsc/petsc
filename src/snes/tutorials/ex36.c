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

  ierr = PetscOptionsBegin(comm, "", "Homogenization Problem Options", "DMPLEX");PetscCall(ierr);
  mod = options->modType;
  PetscCall(PetscOptionsEList("-mod_type", "The model type", "ex36.c", modTypes, NUM_MOD_TYPES, modTypes[options->modType], &mod, NULL));
  options->modType = (ModType) mod;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *user)
{
  PetscBag       bag;
  Parameter     *p;

  PetscFunctionBeginUser;
  PetscCall(PetscBagCreate(comm, sizeof(Parameter), &user->bag));
  PetscCall(PetscBagGetData(user->bag, (void **) &p));
  PetscCall(PetscBagSetName(user->bag, "par", "Homogenization parameters"));
  bag  = user->bag;
  PetscCall(PetscBagRegisterReal(bag, &p->epsilon, 1.0, "epsilon", "Wavelength of fine scale oscillation"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscBagGetData(user->bag, (void **) &ctx));
  switch (user->modType) {
    case MOD_CONSTANT:
      PetscCall(PetscDSSetResidual(ds, 0, f0_trig_homogeneous_u, f1_u));
      PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
      PetscCall(DMGetLabel(dm, "marker", &label));
      ex   = trig_homogeneous_u;
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ex, NULL, ctx, NULL));
      break;
    case MOD_OSCILLATORY:
      PetscCall(PetscDSSetResidual(ds, 0, f0_oscillatory_u, f1_oscillatory_u));
      PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_oscillatory_uu));
      PetscCall(DMGetLabel(dm, "marker", &label));
      ex   = oscillatory_u;
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ex, NULL, ctx, NULL));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_WRONG, "Unsupported model type: %s (%D)", modTypes[PetscMin(user->modType, NUM_MOD_TYPES)], user->modType);
  }
  PetscCall(PetscDSSetExactSolution(ds, 0, ex, ctx));
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[NUM_CONSTANTS];

    PetscCall(PetscBagGetData(user->bag, (void **) &param));

    constants[EPSILON] = param->epsilon;
    PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  /* Create finite element */
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
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
  PetscCall(VecGetDM(u, &dm));
  PetscCall(PetscObjectGetOptions((PetscObject) dm, &options));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix));
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject) dm), options, prefix, "-compare_view", &viewer, &format, &flg));
  if (flg) {
    PetscCall(DMGetGlobalVector(dm, &exact));
    PetscCall(DMComputeExactSolution(dm, 0.0, exact, NULL));
    v[0] = u;
    v[1] = exact;
    for (i = 0; i < 2; ++i) {
      PetscCall(DMGetLocalVector(dm, &lv[i]));
      PetscCall(PetscObjectGetName((PetscObject) v[i], &name));
      PetscCall(PetscObjectSetName((PetscObject) lv[i], name));
      PetscCall(DMGlobalToLocalBegin(dm, v[i], INSERT_VALUES, lv[i]));
      PetscCall(DMGlobalToLocalEnd(dm, v[i], INSERT_VALUES, lv[i]));
      PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, lv[i], 0., NULL, NULL, NULL));
    }
    PetscCall(DMPlexVecView1D(dm, 2, lv, viewer));
    for (i = 0; i < 2; ++i) PetscCall(DMRestoreLocalVector(dm, &lv[i]));
    PetscCall(DMRestoreGlobalVector(dm, &exact));
    PetscCall(PetscViewerDestroy(&viewer));
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
  PetscCall(MatShellGetContext(Pi, (void **) &ctx));
  PetscCall(MatDestroy(&ctx->Mcoarse));
  PetscCall(MatDestroy(&ctx->Mfine));
  PetscCall(MatDestroy(&ctx->Ifine));
  PetscCall(VecDestroy(&ctx->Iscale));
  PetscCall(KSPDestroy(&ctx->kspCoarse));
  PetscCall(VecDestroy(&ctx->tmpcoarse));
  PetscCall(VecDestroy(&ctx->tmpfine));
  PetscCall(PetscFree(ctx));
  PetscCall(MatShellSetContext(Pi, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode CoarseProjection(Mat Pi, Vec x, Vec y)
{
  ProjStruct    *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Pi, (void **) &ctx));
  PetscCall(MatMult(ctx->Mfine, x, ctx->tmpfine));
  PetscCall(PetscObjectSetName((PetscObject) ctx->tmpfine, "Fine DG RHS"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpfine, "fine_dg_"));
  PetscCall(VecViewFromOptions(ctx->tmpfine, NULL, "-rhs_view"));
  PetscCall(MatMultTranspose(ctx->Ifine, ctx->tmpfine, ctx->tmpcoarse));
  PetscCall(PetscObjectSetName((PetscObject) ctx->tmpcoarse, "Coarse DG RHS"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpcoarse, "coarse_dg_"));
  PetscCall(VecViewFromOptions(ctx->tmpcoarse, NULL, "-rhs_view"));
  PetscCall(VecPointwiseMult(ctx->tmpcoarse, ctx->Iscale, ctx->tmpcoarse));
  PetscCall(VecViewFromOptions(ctx->tmpcoarse, NULL, "-rhs_view"));
  PetscCall(KSPSolve(ctx->kspCoarse, ctx->tmpcoarse, y));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateCoarseProjection(DM dmc, DM dmf, Mat *Pi)
{
  ProjStruct    *ctx;
  PetscInt       m, n, M, N;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(1, &ctx));
  PetscCall(DMCreateGlobalVector(dmc, &ctx->tmpcoarse));
  PetscCall(DMCreateGlobalVector(dmf, &ctx->tmpfine));
  PetscCall(VecGetLocalSize(ctx->tmpcoarse, &m));
  PetscCall(VecGetSize(ctx->tmpcoarse, &M));
  PetscCall(VecGetLocalSize(ctx->tmpfine, &n));
  PetscCall(VecGetSize(ctx->tmpfine, &N));
  PetscCall(DMCreateMassMatrix(dmc, dmc, &ctx->Mcoarse));
  PetscCall(DMCreateMassMatrix(dmf, dmf, &ctx->Mfine));
  PetscCall(DMCreateInterpolation(dmc, dmf, &ctx->Ifine, &ctx->Iscale));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject) dmc), &ctx->kspCoarse));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) ctx->kspCoarse, "coarse_"));
  PetscCall(KSPSetOperators(ctx->kspCoarse, ctx->Mcoarse, ctx->Mcoarse));
  PetscCall(KSPSetFromOptions(ctx->kspCoarse));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject) dmc), m, n, M, N, ctx, Pi));
  PetscCall(MatShellSetOperation(*Pi, MATOP_DESTROY, (void (*)(void)) DestroyCoarseProjection));
  PetscCall(MatShellSetOperation(*Pi, MATOP_MULT, (void (*)(void)) CoarseProjection));
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
  PetscCall(MatShellGetContext(P, (void **) &ctx));
  PetscCall(MatDestroy(&ctx->Ifdg));
  PetscCall(MatDestroy(&ctx->Pi));
  PetscCall(VecDestroy(&ctx->tmpc));
  PetscCall(VecDestroy(&ctx->tmpf));
  PetscCall(PetscFree(ctx));
  PetscCall(MatShellSetContext(P, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode QuasiInterpolate(Mat P, Vec x, Vec y)
{
  QuasiInterp   *ctx;
  DM             dmcdg, dmc;
  Vec            ly;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(P, (void **) &ctx));
  PetscCall(MatMult(ctx->Ifdg, x, ctx->tmpf));

  PetscCall(PetscObjectSetName((PetscObject) ctx->tmpf, "Fine DG Potential"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpf, "fine_dg_"));
  PetscCall(VecViewFromOptions(ctx->tmpf, NULL, "-vec_view"));
  PetscCall(MatMult(ctx->Pi, x, ctx->tmpc));

  PetscCall(PetscObjectSetName((PetscObject) ctx->tmpc, "Coarse DG Potential"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpc, "coarse_dg_"));
  PetscCall(VecViewFromOptions(ctx->tmpc, NULL, "-vec_view"));
  PetscCall(VecGetDM(ctx->tmpc, &dmcdg));

  PetscCall(VecGetDM(y, &dmc));
  PetscCall(DMGetLocalVector(dmc, &ly));
  PetscCall(DMPlexComputeClementInterpolant(dmcdg, ctx->tmpc, ly));
  PetscCall(DMLocalToGlobalBegin(dmc, ly, INSERT_VALUES, y));
  PetscCall(DMLocalToGlobalEnd(dmc, ly, INSERT_VALUES, y));
  PetscCall(DMRestoreLocalVector(dmc, &ly));
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
  PetscCall(PetscCalloc1(1, &ctx));
  PetscCall(DMGetGlobalVector(dmc, &x));
  PetscCall(DMGetGlobalVector(dmf, &y));
  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(VecGetSize(x, &M));
  PetscCall(VecGetLocalSize(y, &n));
  PetscCall(VecGetSize(y, &N));
  PetscCall(DMRestoreGlobalVector(dmc, &x));
  PetscCall(DMRestoreGlobalVector(dmf, &y));

  PetscCall(DMClone(dmf, &dmfdg));
  PetscCall(DMGetDimension(dmfdg, &dim));
  PetscCall(DMPlexGetHeightStratum(dmfdg, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dmfdg, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "fine_dg_", PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(dmfdg, 0, NULL, (PetscObject) fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dmfdg));
  PetscCall(DMCreateInterpolation(dmf, dmfdg, &ctx->Ifdg, NULL));
  PetscCall(DMCreateGlobalVector(dmfdg, &ctx->tmpf));
  PetscCall(DMDestroy(&dmfdg));

  PetscCall(DMClone(dmc, &dmcdg));
  PetscCall(DMGetDimension(dmcdg, &dim));
  PetscCall(DMPlexGetHeightStratum(dmcdg, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dmcdg, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "coarse_dg_", PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(dmcdg, 0, NULL, (PetscObject) fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dmcdg));

  PetscCall(CreateCoarseProjection(dmcdg, dmf, &ctx->Pi));
  PetscCall(DMCreateGlobalVector(dmcdg, &ctx->tmpc));
  PetscCall(DMDestroy(&dmcdg));

  PetscCall(MatCreateShell(PetscObjectComm((PetscObject) dmc), m, n, M, N, ctx, P));
  PetscCall(MatShellSetOperation(*P, MATOP_DESTROY, (void (*)(void)) DestroyQuasiInterpolator));
  PetscCall(MatShellSetOperation(*P, MATOP_MULT, (void (*)(void)) QuasiInterpolate));
  PetscFunctionReturn(0);
}

static PetscErrorCode CoarseTest(DM dm, Vec u, AppCtx *user)
{
  DM             dmc;
  Mat            P;    /* The quasi-interpolator to the coarse space */
  Vec            uc;

  PetscFunctionBegin;
  if (user->modType == MOD_CONSTANT) PetscFunctionReturn(0);
  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), &dmc));
  PetscCall(DMSetType(dmc, DMPLEX));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) dmc, "coarse_"));
  PetscCall(DMSetApplicationContext(dmc, user));
  PetscCall(DMSetFromOptions(dmc));
  PetscCall(DMViewFromOptions(dmc, NULL, "-dm_view"));

  PetscCall(SetupDiscretization(dmc, "potential", SetupPrimalProblem, user));
  PetscCall(DMCreateGlobalVector(dmc, &uc));
  PetscCall(PetscObjectSetName((PetscObject) uc, "potential"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) uc, "coarse_"));

  PetscCall(CreateQuasiInterpolator(dmc, dm, &P));
#if 1
  PetscCall(MatMult(P, u, uc));
#else
  {
    Mat In;
    Vec sc;

    PetscCall(DMCreateInterpolation(dmc, dm, &In, &sc));
    PetscCall(MatMultTranspose(In, u, uc));
    PetscCall(VecPointwiseMult(uc, sc, uc));
    PetscCall(MatDestroy(&In));
    PetscCall(VecDestroy(&sc));
  }
#endif
  PetscCall(CompareView(uc));

  PetscCall(MatDestroy(&P));
  PetscCall(VecDestroy(&uc));
  PetscCall(DMDestroy(&dmc));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(SetupParameters(PETSC_COMM_WORLD, &user));
  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject) u, "potential"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetSolution(snes, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-potential_view"));
  PetscCall(CompareView(u));
  /* Looking at a coarse problem */
  PetscCall(CoarseTest(dm, u, &user));
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscBagDestroy(&user.bag));
  PetscCall(PetscFinalize());
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
