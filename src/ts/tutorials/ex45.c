static char help[] = "Heat Equation in 2d and 3d with finite elements.\n\
We solve the heat equation in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
Contributed by: Julian Andrej <juan@tf.uni-kiel.de>\n\n\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

/*
  Heat equation:

    du/dt - \Delta u + f = 0
*/

typedef enum {SOL_QUADRATIC_LINEAR, SOL_QUADRATIC_TRIG, SOL_TRIG_LINEAR, NUM_SOLUTION_TYPES} SolutionType;
const char *solutionTypes[NUM_SOLUTION_TYPES+1] = {"quadratic_linear", "quadratic_trig", "trig_linear", "unknown"};

typedef struct {
  char         filename[PETSC_MAX_PATH_LEN];   /* Mesh filename */
  char         bdfilename[PETSC_MAX_PATH_LEN]; /* Mesh boundary filename */
  PetscReal    scale;                          /* Scale factor for mesh */
  SolutionType solType;                        /* Type of exact solution */
} AppCtx;

/*
Exact 2D solution:
  u = 2t + x^2 + y^2
  F(u) = 2 - (2 + 2) + 2 = 0

Exact 3D solution:
  u = 3t + x^2 + y^2 + z^2
  F(u) = 3 - (2 + 2 + 2) + 3 = 0
*/
static PetscErrorCode mms_quad_lin(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  *u = dim*time;
  for (d = 0; d < dim; ++d) *u += x[d]*x[d];
  return 0;
}

static PetscErrorCode mms_quad_lin_t(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = dim;
  return 0;
}

static void f0_quad_lin(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u_t[0] + (PetscScalar) dim;
}

/*
Exact 2D solution:
  u = 2*cos(t) + x^2 + y^2
  F(u) = -2*sint(t) - (2 + 2) + 2*sin(t) + 4 = 0

Exact 3D solution:
  u = 3*cos(t) + x^2 + y^2 + z^2
  F(u) = -3*sin(t) - (2 + 2 + 2) + 3*sin(t) + 6 = 0
*/
static PetscErrorCode mms_quad_trig(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  *u = dim*PetscCosReal(time);
  for (d = 0; d < dim; ++d) *u += x[d]*x[d];
  return 0;
}

static PetscErrorCode mms_quad_trig_t(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = -dim*PetscSinReal(time);
  return 0;
}

static void f0_quad_trig(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u_t[0] + dim*(PetscSinReal(t) + 2.0);
}

/*
Exact 2D solution:
  u = 2\pi^2 t + cos(\pi x) + cos(\pi y)
  F(u) = 2\pi^2 - \pi^2 (cos(\pi x) + cos(\pi y)) + \pi^2 (cos(\pi x) + cos(\pi y)) - 2\pi^2 = 0

Exact 3D solution:
  u = 3\pi^2 t + cos(\pi x) + cos(\pi y) + cos(\pi z)
  F(u) = 3\pi^2 - \pi^2 (cos(\pi x) + cos(\pi y) + cos(\pi z)) + \pi^2 (cos(\pi x) + cos(\pi y) + cos(\pi z)) - 3\pi^2 = 0
*/
static PetscErrorCode mms_trig_lin(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  *u = dim*PetscSqr(PETSC_PI)*time;
  for (d = 0; d < dim; ++d) *u += PetscCosReal(PETSC_PI*x[d]);
  return 0;
}

static PetscErrorCode mms_trig_lin_t(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = dim*PetscSqr(PETSC_PI);
  return 0;
}

static void f0_trig_lin(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = u_t[0];
  for (d = 0; d < dim; ++d) f0[0] += PetscSqr(PETSC_PI)*(PetscCosReal(PETSC_PI*x[d]) - 1.0);
}

static void f1_temp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_temp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static void g0_temp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u_tShift*1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       sol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0]   = '\0';
  options->bdfilename[0] = '\0';
  options->scale         = 0.0;
  options->solType       = SOL_QUADRATIC_LINEAR;

  ierr = PetscOptionsBegin(comm, "", "Heat Equation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex45.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-bd_filename", "The mesh boundary file", "ex45.c", options->bdfilename, options->bdfilename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-scale", "Scale factor for the mesh", "ex45.c", options->scale, &options->scale, NULL);CHKERRQ(ierr);
  sol  = options->solType;
  ierr = PetscOptionsEList("-sol_type", "Type of exact solution", "ex45.c", solutionTypes, NUM_SOLUTION_TYPES, solutionTypes[options->solType], &sol, NULL);CHKERRQ(ierr);
  options->solType = (SolutionType) sol;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DM             plex;
  DMLabel        label;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMHasLabel(dm, name, &hasLabel);CHKERRQ(ierr);
  if (hasLabel) PetscFunctionReturn(0);
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(plex, 1, label);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  size_t         len, lenbd;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(ctx->filename,   &len);CHKERRQ(ierr);
  ierr = PetscStrlen(ctx->bdfilename, &lenbd);CHKERRQ(ierr);
  if (lenbd) {
    DM bdm;

    ierr = DMPlexCreateFromFile(comm, ctx->bdfilename, PETSC_TRUE, &bdm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) bdm, "bd_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(bdm);CHKERRQ(ierr);
    if (ctx->scale != 0.0) {
      Vec coordinates, coordinatesLocal;

      ierr = DMGetCoordinates(bdm, &coordinates);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(bdm, &coordinatesLocal);CHKERRQ(ierr);
      ierr = VecScale(coordinates, ctx->scale);CHKERRQ(ierr);
      ierr = VecScale(coordinatesLocal, ctx->scale);CHKERRQ(ierr);
    }
    ierr = DMViewFromOptions(bdm, NULL, "-dm_view");CHKERRQ(ierr);
    ierr = DMPlexGenerate(bdm, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMDestroy(&bdm);CHKERRQ(ierr);
  } else if (len) {
    ierr = DMPlexCreateFromFile(comm, ctx->filename, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);
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
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, g0_temp, NULL, NULL, g3_temp);CHKERRQ(ierr);
  switch (ctx->solType) {
    case SOL_QUADRATIC_LINEAR:
      ierr = PetscDSSetResidual(ds, 0, f0_quad_lin,  f1_temp);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(ds, 0, mms_quad_lin, ctx);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolutionTimeDerivative(ds, 0, mms_quad_lin_t, ctx);CHKERRQ(ierr);
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) mms_quad_lin, (void (*)(void)) mms_quad_lin_t, ctx, NULL);CHKERRQ(ierr);
      break;
    case SOL_QUADRATIC_TRIG:
      ierr = PetscDSSetResidual(ds, 0, f0_quad_trig, f1_temp);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(ds, 0, mms_quad_trig, ctx);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolutionTimeDerivative(ds, 0, mms_quad_trig_t, ctx);CHKERRQ(ierr);
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) mms_quad_trig, (void (*)(void)) mms_quad_trig_t, ctx, NULL);CHKERRQ(ierr);
      break;
    case SOL_TRIG_LINEAR:
      ierr = PetscDSSetResidual(ds, 0, f0_trig_lin,  f1_temp);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolution(ds, 0, mms_trig_lin, ctx);CHKERRQ(ierr);
      ierr = PetscDSSetExactSolutionTimeDerivative(ds, 0, mms_trig_lin_t, ctx);CHKERRQ(ierr);
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) mms_trig_lin, (void (*)(void)) mms_trig_lin_t, ctx, NULL);CHKERRQ(ierr);
      break;
    default: SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Invalid solution type: %s (%D)", solutionTypes[PetscMin(ctx->solType, NUM_SOLUTION_TYPES)], ctx->solType);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx* ctx)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "temp_", -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "temperature");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, ctx);CHKERRQ(ierr);
  while (cdm) {
    ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
  DM             dm;
  PetscReal      t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMComputeExactSolution(dm, t, u, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  TS             ts;
  Vec            u;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &ctx);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
  ierr = SetInitialConditions(ts, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "temperature");CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p1
    requires: triangle
    args: -sol_type quadratic_linear -dm_refine 1 -temp_petscspace_degree 1 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [1.9]
    suffix: 2d_p1_sconv
    requires: triangle
    args: -sol_type quadratic_linear -temp_petscspace_degree 1 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 4 -convest_num_refine 3 get L_2 convergence rate: [1.2]
    suffix: 2d_p1_tconv
    requires: triangle
    args: -sol_type quadratic_trig -temp_petscspace_degree 1 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    suffix: 2d_p2
    requires: triangle
    args: -sol_type quadratic_linear -dm_refine 0 -temp_petscspace_degree 2 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [2.9]
    suffix: 2d_p2_sconv
    requires: triangle
    args: -sol_type trig_linear -temp_petscspace_degree 2 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00000001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 3 -convest_num_refine 3 get L_2 convergence rate: [1.0]
    suffix: 2d_p2_tconv
    requires: triangle
    args: -sol_type quadratic_trig -temp_petscspace_degree 2 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    suffix: 2d_q1
    args: -sol_type quadratic_linear -dm_plex_box_simplex 0 -dm_refine 1 -temp_petscspace_degree 1 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [1.9]
    suffix: 2d_q1_sconv
    args: -sol_type quadratic_linear -dm_plex_box_simplex 0 -temp_petscspace_degree 1 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 4 -convest_num_refine 3 get L_2 convergence rate: [1.2]
    suffix: 2d_q1_tconv
    args: -sol_type quadratic_trig -dm_plex_box_simplex 0 -temp_petscspace_degree 1 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    suffix: 2d_q2
    args: -sol_type quadratic_linear -dm_plex_box_simplex 0 -dm_refine 0 -temp_petscspace_degree 2 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [2.9]
    suffix: 2d_q2_sconv
    args: -sol_type trig_linear -dm_plex_box_simplex 0 -temp_petscspace_degree 2 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00000001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 3 -convest_num_refine 3 get L_2 convergence rate: [1.0]
    suffix: 2d_q2_tconv
    args: -sol_type quadratic_trig -dm_plex_box_simplex 0 -temp_petscspace_degree 2 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu

  test:
    suffix: 3d_p1
    requires: ctetgen
    args: -sol_type quadratic_linear -dm_refine 1 -temp_petscspace_degree 1 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [1.9]
    suffix: 3d_p1_sconv
    requires: ctetgen
    args: -sol_type quadratic_linear -temp_petscspace_degree 1 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 4 -convest_num_refine 3 get L_2 convergence rate: [1.2]
    suffix: 3d_p1_tconv
    requires: ctetgen
    args: -sol_type quadratic_trig -temp_petscspace_degree 1 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    suffix: 3d_p2
    requires: ctetgen
    args: -sol_type quadratic_linear -dm_refine 0 -temp_petscspace_degree 2 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [2.9]
    suffix: 3d_p2_sconv
    requires: ctetgen
    args: -sol_type trig_linear -temp_petscspace_degree 2 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00000001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 3 -convest_num_refine 3 get L_2 convergence rate: [1.0]
    suffix: 3d_p2_tconv
    requires: ctetgen
    args: -sol_type quadratic_trig -temp_petscspace_degree 2 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    suffix: 3d_q1
    args: -sol_type quadratic_linear -dm_plex_box_simplex 0 -dm_refine 1 -temp_petscspace_degree 1 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [1.9]
    suffix: 3d_q1_sconv
    args: -sol_type quadratic_linear -dm_plex_box_simplex 0 -temp_petscspace_degree 1 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 4 -convest_num_refine 3 get L_2 convergence rate: [1.2]
    suffix: 3d_q1_tconv
    args: -sol_type quadratic_trig -dm_plex_box_simplex 0 -temp_petscspace_degree 1 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    suffix: 3d_q2
    args: -sol_type quadratic_linear -dm_plex_box_simplex 0 -dm_refine 0 -temp_petscspace_degree 2 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 2 -convest_num_refine 3 get L_2 convergence rate: [2.9]
    suffix: 3d_q2_sconv
    args: -sol_type trig_linear -dm_plex_box_simplex 0 -temp_petscspace_degree 2 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 1 -ts_dt 0.00000001 -snes_error_if_not_converged -pc_type lu
  test:
    # -dm_refine 3 -convest_num_refine 3 get L_2 convergence rate: [1.0]
    suffix: 3d_q2_tconv
    args: -sol_type quadratic_trig -dm_plex_box_simplex 0 -temp_petscspace_degree 2 -ts_convergence_estimate -convest_num_refine 1 \
          -ts_type beuler -ts_max_steps 4 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu

  test:
    # For a nice picture, -bd_dm_refine 2 -dm_refine 1 -dm_view hdf5:${PETSC_DIR}/sol.h5 -ts_monitor_solution hdf5:${PETSC_DIR}/sol.h5::append
    suffix: egads_sphere
    requires: egads ctetgen
    args: -sol_type quadratic_linear -bd_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/unit_sphere.egadslite -scale 40 \
          -temp_petscspace_degree 2 -dmts_check .0001 \
          -ts_type beuler -ts_max_steps 5 -ts_dt 0.1 -snes_error_if_not_converged -pc_type lu

TEST*/
