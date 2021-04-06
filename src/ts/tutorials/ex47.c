static char help[] = "Pure advection with finite elements.\n\
We solve the hyperbolic problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*
The continuity equation (https://en.wikipedia.org/wiki/Continuity_equation) for advection
(https://en.wikipedia.org/wiki/Advection) of a conserved scalar quantity phi, with source q,

  phi_t + div (phi u) = q

if used with a solenoidal velocity field u (div u = 0) is given by

  phi_t + u . grad phi = q

For a vector quantity a, we likewise have

  a_t + u . grad a = q
*/

/*
  r1: 8 SOR
  r2: 1128 SOR
  r3: > 10000 SOR

  SOR is completely unreliable as a smoother, use Jacobi
  r1: 8 MG
  r2:
*/

#include <petscdmplex.h>
#include <petscts.h>
#include <petscds.h>

typedef enum {PRIMITIVE, INT_BY_PARTS} WeakFormType;

typedef struct {
  /* Domain and mesh definition */
  PetscInt          dim;               /* The topological mesh dimension */
  PetscBool         simplex;           /* Use simplices or tensor product cells */
  /* Problem definition */
  WeakFormType      formType;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

/* MMS1:

  2D:
  u   = <1, 1>
  phi = x + y - 2t

  phi_t + u . grad phi = -2 + <1, 1> . <1, 1> = 0

  3D:
  u   = <1, 1, 1>
  phi = x + y + z - 3t

  phi_t + u . grad phi = -3 + <1, 1, 1> . <1, 1, 1> = 0
*/

static PetscErrorCode analytic_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;

  *u = -dim*time;
  for (d = 0; d < dim; ++d) *u += x[d];
  return 0;
}

static PetscErrorCode velocity(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 1.0;
  return 0;
}

/* <psi, phi_t> + <psi, u . grad phi> */
static void f0_prim_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  f0[0] = u_t[0];
  for (d = 0; d < dim; ++d) f0[0] += a[d] * u_x[d];
}

/* <psi, phi_t> */
static void f0_ibp_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u_t[0];
}

/* <grad psi, u phi> */
static void f1_ibp_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = a[d]*u[0];
}

/* <psi, phi_t> */
static void g0_prim_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift*1.0;
}

/* <psi, u . grad phi> */
static void g1_prim_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d] = a[d];
}

/* <grad psi, u phi> */
static void g2_ibp_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d] = a[d];
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *formTypes[2] = {"primitive", "int_by_parts"};
  PetscInt       form;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim      = 2;
  options->simplex  = PETSC_TRUE;
  options->formType = PRIMITIVE;

  ierr = PetscOptionsBegin(comm, "", "Advection Equation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex47.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex47.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  form = options->formType;
  ierr = PetscOptionsEList("-form_type", "The weak form type", "ex47.c", formTypes, 2, formTypes[options->formType], &form, NULL);CHKERRQ(ierr);
  options->formType = (WeakFormType) form;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DM             plex;
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  DM             pdm = NULL;
  const PetscInt dim = ctx->dim;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, dim, ctx->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  /* If no boundary marker exists, mark the whole boundary */
  ierr = DMHasLabel(*dm, "marker", &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) {ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);}
  /* Distribute mesh over processes */
  ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
  if (pdm) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = pdm;
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *ctx)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  switch (ctx->formType) {
  case PRIMITIVE:
    ierr = PetscDSSetResidual(ds, 0, f0_prim_phi, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds, 0, 0, g0_prim_phi, g1_prim_phi, NULL, NULL);CHKERRQ(ierr);
    break;
  case INT_BY_PARTS:
    ierr = PetscDSSetResidual(ds, 0, f0_ibp_phi, f1_ibp_phi);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds, 0, 0, g0_prim_phi, NULL, g2_ibp_phi, NULL);CHKERRQ(ierr);
    break;
  }
  ctx->exactFuncs[0] = analytic_phi;
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) ctx->exactFuncs[0], NULL, ctx, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupVelocity(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {velocity};
  Vec            v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLocalVector(dmAux, &v);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, funcs, NULL, INSERT_ALL_VALUES, v);CHKERRQ(ierr);
  ierr = DMSetAuxiliaryVec(dm, NULL, 0, v);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscFE feAux, AppCtx *user)
{
  DM             dmAux, coordDM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  if (!feAux) PetscFunctionReturn(0);
  ierr = DMClone(dm, &dmAux);CHKERRQ(ierr);
  ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
  ierr = DMSetField(dmAux, 0, NULL, (PetscObject) feAux);CHKERRQ(ierr);
  ierr = DMCreateDS(dmAux);CHKERRQ(ierr);
  ierr = SetupVelocity(dm, dmAux, user);CHKERRQ(ierr);
  ierr = DMDestroy(&dmAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx* ctx)
{
  DM              cdm = dm;
  const PetscInt  dim = ctx->dim;
  PetscFE         fe,   feAux;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, ctx->simplex, "phi_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "phi");CHKERRQ(ierr);
  /* Create velocity */
  ierr = PetscFECreateDefault(comm, dim, dim, ctx->simplex, "vel_", -1, &feAux);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe, feAux);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, ctx);CHKERRQ(ierr);
  while (cdm) {
    PetscBool hasLabel;

    ierr = SetupAuxDM(cdm, feAux, ctx);CHKERRQ(ierr);
    ierr = DMHasLabel(cdm, "marker", &hasLabel);CHKERRQ(ierr);
    if (!hasLabel) {ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);}
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feAux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(KSP ksp, PetscInt it, PetscReal rnorm, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  Vec            u, r, error;
  PetscReal      time = 0.5, res;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, it, time);CHKERRQ(ierr);
  /* Calculate residual */
  ierr = KSPBuildResidual(ksp, NULL, NULL, &r);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, it, res);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "residual");CHKERRQ(ierr);
  ierr = VecViewFromOptions(r, NULL, "-res_vec_view");CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  /* Calculate error */
  ierr = KSPBuildSolution(ksp, NULL, &u);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &error);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, time, user->exactFuncs, NULL, INSERT_ALL_VALUES, error);CHKERRQ(ierr);
  ierr = VecAXPY(error, -1.0, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) error, "error");CHKERRQ(ierr);
  ierr = VecViewFromOptions(error, NULL, "-err_vec_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &error);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyTSMonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  PetscReal      error;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, crtime, user->exactFuncs, NULL, u, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: %2.5g\n", (int) step, (double) crtime, (double) error);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         ctx;
  DM             dm;
  TS             ts;
  Vec            u, r;
  PetscReal      t       = 0.0;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &ctx.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &ctx);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "phi");CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts, MyTSMonitorError, &ctx, NULL);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, t, ctx.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  {
    SNES snes;
    KSP  ksp;

    ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp, MonitorError, &ctx, NULL);CHKERRQ(ierr);
  }
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(ctx.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # Full solves
  test:
    suffix: 2d_p1p1_r1
    requires: triangle
    args: -dm_refine 1 -phi_petscspace_degree 1 -vel_petscspace_degree 1 -ts_type beuler -ts_max_steps 10 -ts_dt 0.1 -pc_type lu -snes_monitor_short -snes_converged_reason -ts_monitor

  test:
    suffix: 2d_p1p1_sor_r1
    requires: triangle !single
    args: -dm_refine 1 -phi_petscspace_degree 1 -vel_petscspace_degree 1 -ts_type beuler -ts_max_steps 10 -ts_dt 0.1 -ksp_rtol 1.0e-9 -pc_type sor -snes_monitor_short -snes_converged_reason -ksp_monitor_short -ts_monitor

  test:
    suffix: 2d_p1p1_mg_r1
    requires: triangle !single
    args: -dm_refine_hierarchy 1 -phi_petscspace_degree 1 -vel_petscspace_degree 1 -ts_type beuler -ts_max_steps 10 -ts_dt 0.1 -ksp_type fgmres -ksp_rtol 1.0e-9 -pc_type mg -pc_mg_levels 2 -snes_monitor_short -snes_converged_reason -snes_view -ksp_monitor_true_residual -ts_monitor

TEST*/
