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
  WeakFormType formType;
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
  options->formType = PRIMITIVE;

  ierr = PetscOptionsBegin(comm, "", "Advection Equation Options", "DMPLEX");CHKERRQ(ierr);
  form = options->formType;
  CHKERRQ(PetscOptionsEList("-form_type", "The weak form type", "ex47.c", formTypes, 2, formTypes[options->formType], &form, NULL));
  options->formType = (WeakFormType) form;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *ctx)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &ds));
  switch (ctx->formType) {
  case PRIMITIVE:
    CHKERRQ(PetscDSSetResidual(ds, 0, f0_prim_phi, NULL));
    CHKERRQ(PetscDSSetJacobian(ds, 0, 0, g0_prim_phi, g1_prim_phi, NULL, NULL));
    break;
  case INT_BY_PARTS:
    CHKERRQ(PetscDSSetResidual(ds, 0, f0_ibp_phi, f1_ibp_phi));
    CHKERRQ(PetscDSSetJacobian(ds, 0, 0, g0_prim_phi, NULL, g2_ibp_phi, NULL));
    break;
  }
  CHKERRQ(PetscDSSetExactSolution(ds, 0, analytic_phi, ctx));
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) analytic_phi, NULL, ctx, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupVelocity(DM dm, DM dmAux, AppCtx *user)
{
  PetscSimplePointFunc funcs[1] = {velocity};
  Vec                  v;

  PetscFunctionBeginUser;
  CHKERRQ(DMCreateLocalVector(dmAux, &v));
  CHKERRQ(DMProjectFunctionLocal(dmAux, 0.0, funcs, NULL, INSERT_ALL_VALUES, v));
  CHKERRQ(DMSetAuxiliaryVec(dm, NULL, 0, 0, v));
  CHKERRQ(VecDestroy(&v));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscFE feAux, AppCtx *user)
{
  DM             dmAux, coordDM;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  CHKERRQ(DMGetCoordinateDM(dm, &coordDM));
  if (!feAux) PetscFunctionReturn(0);
  CHKERRQ(DMClone(dm, &dmAux));
  CHKERRQ(DMSetCoordinateDM(dmAux, coordDM));
  CHKERRQ(DMSetField(dmAux, 0, NULL, (PetscObject) feAux));
  CHKERRQ(DMCreateDS(dmAux));
  CHKERRQ(SetupVelocity(dm, dmAux, user));
  CHKERRQ(DMDestroy(&dmAux));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx* ctx)
{
  DM             cdm = dm;
  PetscFE        fe,   feAux;
  MPI_Comm       comm;
  PetscInt       dim;
  PetscBool      simplex;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(PetscFECreateDefault(comm, dim, 1, simplex, "phi_", -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, "phi"));
  CHKERRQ(PetscFECreateDefault(comm, dim, dim, simplex, "vel_", -1, &feAux));
  CHKERRQ(PetscFECopyQuadrature(fe, feAux));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(SetupProblem(dm, ctx));
  while (cdm) {
    CHKERRQ(SetupAuxDM(cdm, feAux, ctx));
    CHKERRQ(DMCopyDisc(dm, cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(PetscFEDestroy(&feAux));
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(KSP ksp, PetscInt it, PetscReal rnorm, void *ctx)
{
  DM                   dm;
  PetscDS              ds;
  PetscSimplePointFunc func[1];
  void                *ctxs[1];
  Vec                  u, r, error;
  PetscReal            time = 0.5, res;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp, &dm));
  CHKERRQ(DMSetOutputSequenceNumber(dm, it, time));
  /* Calculate residual */
  CHKERRQ(KSPBuildResidual(ksp, NULL, NULL, &r));
  CHKERRQ(VecNorm(r, NORM_2, &res));
  CHKERRQ(DMSetOutputSequenceNumber(dm, it, res));
  CHKERRQ(PetscObjectSetName((PetscObject) r, "residual"));
  CHKERRQ(VecViewFromOptions(r, NULL, "-res_vec_view"));
  CHKERRQ(VecDestroy(&r));
  /* Calculate error */
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSGetExactSolution(ds, 0, &func[0], &ctxs[0]));
  CHKERRQ(KSPBuildSolution(ksp, NULL, &u));
  CHKERRQ(DMGetGlobalVector(dm, &error));
  CHKERRQ(DMProjectFunction(dm, time, func, ctxs, INSERT_ALL_VALUES, error));
  CHKERRQ(VecAXPY(error, -1.0, u));
  CHKERRQ(PetscObjectSetName((PetscObject) error, "error"));
  CHKERRQ(VecViewFromOptions(error, NULL, "-err_vec_view"));
  CHKERRQ(DMRestoreGlobalVector(dm, &error));
  PetscFunctionReturn(0);
}

static PetscErrorCode MyTSMonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
  DM                   dm;
  PetscDS              ds;
  PetscSimplePointFunc func[1];
  void                *ctxs[1];
  PetscReal            error;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSGetExactSolution(ds, 0, &func[0], &ctxs[0]));
  CHKERRQ(DMComputeL2Diff(dm, crtime, func, ctxs, u, &error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: %2.5g\n", (int) step, (double) crtime, (double) error));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         ctx;
  DM             dm;
  TS             ts;
  Vec            u, r;
  PetscReal      t       = 0.0;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &dm, &ctx));
  CHKERRQ(DMSetApplicationContext(dm, &ctx));
  CHKERRQ(SetupDiscretization(dm, &ctx));

  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "phi"));
  CHKERRQ(VecDuplicate(u, &r));

  CHKERRQ(TSCreate(PETSC_COMM_WORLD, &ts));
  CHKERRQ(TSMonitorSet(ts, MyTSMonitorError, &ctx, NULL));
  CHKERRQ(TSSetDM(ts, dm));
  CHKERRQ(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx));
  CHKERRQ(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx));
  CHKERRQ(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx));
  CHKERRQ(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetFromOptions(ts));

  {
    PetscDS              ds;
    PetscSimplePointFunc func[1];
    void                *ctxs[1];

    CHKERRQ(DMGetDS(dm, &ds));
    CHKERRQ(PetscDSGetExactSolution(ds, 0, &func[0], &ctxs[0]));
    CHKERRQ(DMProjectFunction(dm, t, func, ctxs, INSERT_ALL_VALUES, u));
  }
  {
    SNES snes;
    KSP  ksp;

    CHKERRQ(TSGetSNES(ts, &snes));
    CHKERRQ(SNESGetKSP(snes, &ksp));
    CHKERRQ(KSPMonitorSet(ksp, MonitorError, &ctx, NULL));
  }
  CHKERRQ(TSSolve(ts, u));
  CHKERRQ(VecViewFromOptions(u, NULL, "-sol_vec_view"));

  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
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
