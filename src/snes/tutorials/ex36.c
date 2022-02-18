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
  ierr = PetscOptionsEList("-mod_type", "The model type", "ex36.c", modTypes, NUM_MOD_TYPES, modTypes[options->modType], &mod, NULL);CHKERRQ(ierr);
  options->modType = (ModType) mod;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *user)
{
  PetscBag       bag;
  Parameter     *p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscBagCreate(comm, sizeof(Parameter), &user->bag);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &p);CHKERRQ(ierr);
  ierr = PetscBagSetName(user->bag, "par", "Homogenization parameters");CHKERRQ(ierr);
  bag  = user->bag;
  ierr = PetscBagRegisterReal(bag, &p->epsilon, 1.0, "epsilon", "Wavelength of fine scale oscillation");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS              ds;
  DMLabel              label;
  PetscSimplePointFunc ex;
  const PetscInt       id = 1;
  void                *ctx;
  PetscErrorCode       ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &ctx);CHKERRQ(ierr);
  switch (user->modType) {
    case MOD_CONSTANT:
      ierr = PetscDSSetResidual(ds, 0, f0_trig_homogeneous_u, f1_u);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
      ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
      ex   = trig_homogeneous_u;
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ex, NULL, ctx, NULL);CHKERRQ(ierr);
      break;
    case MOD_OSCILLATORY:
      ierr = PetscDSSetResidual(ds, 0, f0_oscillatory_u, f1_oscillatory_u);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_oscillatory_uu);CHKERRQ(ierr);
      ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
      ex   = oscillatory_u;
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ex, NULL, ctx, NULL);CHKERRQ(ierr);
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) ds), PETSC_ERR_ARG_WRONG, "Unsupported model type: %s (%D)", modTypes[PetscMin(user->modType, NUM_MOD_TYPES)], user->modType);
  }
  ierr = PetscDSSetExactSolution(ds, 0, ex, ctx);CHKERRQ(ierr);
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[NUM_CONSTANTS];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

    constants[EPSILON] = param->epsilon;
    ierr = PetscDSSetConstants(ds, NUM_CONSTANTS, constants);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(u, &dm);CHKERRQ(ierr);
  ierr = PetscObjectGetOptions((PetscObject) dm, &options);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) dm), options, prefix, "-compare_view", &viewer, &format, &flg);
  if (flg) {
    ierr = DMGetGlobalVector(dm, &exact);CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, 0.0, exact, NULL);CHKERRQ(ierr);
    v[0] = u;
    v[1] = exact;
    for (i = 0; i < 2; ++i) {
      ierr = DMGetLocalVector(dm, &lv[i]);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) v[i], &name);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) lv[i], name);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dm, v[i], INSERT_VALUES, lv[i]);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dm, v[i], INSERT_VALUES, lv[i]);CHKERRQ(ierr);
      ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, lv[i], 0., NULL, NULL, NULL);CHKERRQ(ierr);
    }
    ierr = DMPlexVecView1D(dm, 2, lv, viewer);CHKERRQ(ierr);
    for (i = 0; i < 2; ++i) {ierr = DMRestoreLocalVector(dm, &lv[i]);CHKERRQ(ierr);}
    ierr = DMRestoreGlobalVector(dm, &exact);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Pi, (void **) &ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Mcoarse);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Mfine);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Ifine);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->Iscale);CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->kspCoarse);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->tmpcoarse);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->tmpfine);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = MatShellSetContext(Pi, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CoarseProjection(Mat Pi, Vec x, Vec y)
{
  ProjStruct    *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Pi, (void **) &ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->Mfine, x, ctx->tmpfine);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) ctx->tmpfine, "Fine DG RHS");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpfine, "fine_dg_");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ctx->tmpfine, NULL, "-rhs_view");CHKERRQ(ierr);
  ierr = MatMultTranspose(ctx->Ifine, ctx->tmpfine, ctx->tmpcoarse);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) ctx->tmpcoarse, "Coarse DG RHS");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpcoarse, "coarse_dg_");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ctx->tmpcoarse, NULL, "-rhs_view");CHKERRQ(ierr);
  ierr = VecPointwiseMult(ctx->tmpcoarse, ctx->Iscale, ctx->tmpcoarse);CHKERRQ(ierr);
  ierr = VecViewFromOptions(ctx->tmpcoarse, NULL, "-rhs_view");CHKERRQ(ierr);
  ierr = KSPSolve(ctx->kspCoarse, ctx->tmpcoarse, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateCoarseProjection(DM dmc, DM dmf, Mat *Pi)
{
  ProjStruct    *ctx;
  PetscInt       m, n, M, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(1, &ctx);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmc, &ctx->tmpcoarse);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmf, &ctx->tmpfine);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ctx->tmpcoarse, &m);CHKERRQ(ierr);
  ierr = VecGetSize(ctx->tmpcoarse, &M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(ctx->tmpfine, &n);CHKERRQ(ierr);
  ierr = VecGetSize(ctx->tmpfine, &N);CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(dmc, dmc, &ctx->Mcoarse);CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(dmf, dmf, &ctx->Mfine);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dmc, dmf, &ctx->Ifine, &ctx->Iscale);CHKERRQ(ierr);
  ierr = KSPCreate(PetscObjectComm((PetscObject) dmc), &ctx->kspCoarse);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ctx->kspCoarse, "coarse_");CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->kspCoarse, ctx->Mcoarse, ctx->Mcoarse);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ctx->kspCoarse);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject) dmc), m, n, M, N, ctx, Pi);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Pi, MATOP_DESTROY, (void (*)(void)) DestroyCoarseProjection);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Pi, MATOP_MULT, (void (*)(void)) CoarseProjection);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(P, (void **) &ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Ifdg);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Pi);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->tmpc);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->tmpf);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = MatShellSetContext(P, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode QuasiInterpolate(Mat P, Vec x, Vec y)
{
  QuasiInterp   *ctx;
  DM             dmcdg, dmc;
  Vec            ly;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(P, (void **) &ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->Ifdg, x, ctx->tmpf);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) ctx->tmpf, "Fine DG Potential");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpf, "fine_dg_");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ctx->tmpf, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = MatMult(ctx->Pi, x, ctx->tmpc);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) ctx->tmpc, "Coarse DG Potential");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ctx->tmpc, "coarse_dg_");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ctx->tmpc, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecGetDM(ctx->tmpc, &dmcdg);CHKERRQ(ierr);

  ierr = VecGetDM(y, &dmc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmc, &ly);CHKERRQ(ierr);
  ierr = DMPlexComputeClementInterpolant(dmcdg, ctx->tmpc, ly);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmc, ly, INSERT_VALUES, y);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmc, ly, INSERT_VALUES, y);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmc, &ly);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(1, &ctx);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmc, &x);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmf, &y);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &m);CHKERRQ(ierr);
  ierr = VecGetSize(x, &M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y, &n);CHKERRQ(ierr);
  ierr = VecGetSize(y, &N);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmc, &x);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmf, &y);CHKERRQ(ierr);

  ierr = DMClone(dmf, &dmfdg);CHKERRQ(ierr);
  ierr = DMGetDimension(dmfdg, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmfdg, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dmfdg, cStart, &ct);CHKERRQ(ierr);
  ierr = PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "fine_dg_", PETSC_DETERMINE, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dmfdg, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dmfdg);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dmf, dmfdg, &ctx->Ifdg, NULL);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmfdg, &ctx->tmpf);CHKERRQ(ierr);
  ierr = DMDestroy(&dmfdg);CHKERRQ(ierr);

  ierr = DMClone(dmc, &dmcdg);CHKERRQ(ierr);
  ierr = DMGetDimension(dmcdg, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dmcdg, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dmcdg, cStart, &ct);CHKERRQ(ierr);
  ierr = PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "coarse_dg_", PETSC_DETERMINE, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dmcdg, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dmcdg);CHKERRQ(ierr);

  ierr = CreateCoarseProjection(dmcdg, dmf, &ctx->Pi);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmcdg, &ctx->tmpc);CHKERRQ(ierr);
  ierr = DMDestroy(&dmcdg);CHKERRQ(ierr);

  ierr = MatCreateShell(PetscObjectComm((PetscObject) dmc), m, n, M, N, ctx, P);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*P, MATOP_DESTROY, (void (*)(void)) DestroyQuasiInterpolator);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*P, MATOP_MULT, (void (*)(void)) QuasiInterpolate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CoarseTest(DM dm, Vec u, AppCtx *user)
{
  DM             dmc;
  Mat            P;    /* The quasi-interpolator to the coarse space */
  Vec            uc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (user->modType == MOD_CONSTANT) PetscFunctionReturn(0);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &dmc);CHKERRQ(ierr);
  ierr = DMSetType(dmc, DMPLEX);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dmc, "coarse_");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dmc, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmc);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmc, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = SetupDiscretization(dmc, "potential", SetupPrimalProblem, user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmc, &uc);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uc, "potential");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) uc, "coarse_");CHKERRQ(ierr);

  ierr = CreateQuasiInterpolator(dmc, dm, &P);CHKERRQ(ierr);
#if 1
  ierr = MatMult(P, u, uc);CHKERRQ(ierr);
#else
  {
    Mat In;
    Vec sc;

    ierr = DMCreateInterpolation(dmc, dm, &In, &sc);CHKERRQ(ierr);
    ierr = MatMultTranspose(In, u, uc);CHKERRQ(ierr);
    ierr = VecPointwiseMult(uc, sc, uc);CHKERRQ(ierr);
    ierr = MatDestroy(&In);CHKERRQ(ierr);
    ierr = VecDestroy(&sc);CHKERRQ(ierr);
  }
#endif
  ierr = CompareView(uc);CHKERRQ(ierr);

  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = VecDestroy(&uc);CHKERRQ(ierr);
  ierr = DMDestroy(&dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SetupParameters(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
  ierr = CompareView(u);CHKERRQ(ierr);
  /* Looking at a coarse problem */
  ierr = CoarseTest(dm, u, &user);CHKERRQ(ierr);
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
