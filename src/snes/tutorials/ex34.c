static char help[] = "Poisson problem in 2d with finite elements and bound constraints.\n\
This example is intended to test VI solvers.\n\n\n";

/*
  This is the ball obstacle problem, taken from the MS thesis ``Adaptive Mesh Refinement for Variational Inequalities''
  by Stefano Fochesatto, University of Alaska Fairbanks, 2025
  This is the same VI problem as in src/snes/tutorials/ex9.c, which uses DMDA.  The example
  is also documented by Chapter 12 of E. Bueler, "PETSc for Partial Differential Equations",
  SIAM Press 2021.

  To visualize the solution, configure with petsc4py, pip install pyvista, and use

    -potential_view pyvista  -view_pyvista_warp 1.

  To look at the error use

    -snes_convergence_estimate -convest_num_refine 2 -convest_monitor -convest_error_view pyvista

  and for the inactive residual and active set use

    -snes_vi_monitor_residual pyvista -snes_vi_monitor_active pyvista

  To see the convergence history use

  -snes_vi_monitor -snes_converged_reason -convest_monitor
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscconvest.h>

typedef enum {
  OBSTACLE_NONE,
  OBSTACLE_BALL,
  NUM_OBSTACLE_TYPES
} ObstacleType;
const char *obstacleTypes[NUM_OBSTACLE_TYPES + 1] = {"none", "ball", "unknown"};

typedef struct {
  PetscReal r_0;    // Ball radius
  PetscReal r_free; // Radius of the free boundary for the ball obstacle
  PetscReal A;      // Logarithmic coefficient in exact ball solution
  PetscReal B;      // Constant coefficient in exact ball solution
} Parameter;

typedef struct {
  // Problem definition
  ObstacleType obsType; // Type of obstacle
  PetscBag     bag;     // Problem parameters
} AppCtx;

static PetscErrorCode obstacle_ball(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  Parameter      *par    = (Parameter *)ctx;
  const PetscReal r_0    = par->r_0;
  const PetscReal r      = PetscSqrtReal(PetscSqr(x[0]) + PetscSqr(x[1]));
  const PetscReal psi_0  = PetscSqrtReal(1. - PetscSqr(r_0));
  const PetscReal dpsi_0 = -r_0 / psi_0;

  PetscFunctionBegin;
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Ball obstacle is only defined in 2D");
  if (r < r_0) u[0] = PetscSqrtReal(1.0 - PetscSqr(r));
  else u[0] = psi_0 + dpsi_0 * (r - r_0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode exactSol_ball(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  Parameter      *par    = (Parameter *)ctx;
  const PetscReal r_free = par->r_free;
  const PetscReal A      = par->A;
  const PetscReal B      = par->B;
  const PetscReal r      = PetscSqrtReal(PetscSqr(x[0]) + PetscSqr(x[1]));

  PetscFunctionBegin;
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Ball obstacle is only defined in 2D");
  if (r < r_free) PetscCall(obstacle_ball(dim, time, x, Nc, u, ctx));
  else u[0] = -A * PetscLogReal(r) + B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt obs = OBSTACLE_BALL;

  options->obsType = OBSTACLE_BALL;

  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Ball Obstacle Problem Options", "DMPLEX");
  PetscCall(PetscOptionsEList("-obs_type", "Type of obstacle", "ex34.c", obstacleTypes, NUM_OBSTACLE_TYPES, obstacleTypes[options->obsType], &obs, NULL));
  options->obsType = (ObstacleType)obs;
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *ctx)
{
  PetscBag   bag;
  Parameter *p;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  PetscCall(PetscBagGetData(ctx->bag, (void **)&p));
  PetscCall(PetscBagSetName(ctx->bag, "par", "Obstacle Parameters"));
  bag = ctx->bag;
  PetscCall(PetscBagRegisterReal(bag, &p->r_0, 0.9, "r_0", "Ball radius, m"));
  PetscCall(PetscBagRegisterReal(bag, &p->r_free, 0.697965148223374, "r_free", "Ball free boundary radius, m"));
  PetscCall(PetscBagRegisterReal(bag, &p->A, 0.680259411891719, "A", "Logarithmic coefficient in exact ball solution"));
  PetscCall(PetscBagRegisterReal(bag, &p->B, 0.471519893402112, "B", "Constant coefficient in exact ball solution"));
  PetscCall(PetscBagSetFromOptions(bag));
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;

    PetscCall(PetscOptionsCreateViewer(comm, NULL, NULL, "-param_view", &viewer, &format, &flg));
    if (flg) {
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(PetscBagView(bag, viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscCall(DMGetCoordinatesLocalSetUp(*dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscErrorCode (*exact)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  Parameter    *param;
  PetscDS       ds;
  PetscWeakForm wf;
  DMLabel       label;
  PetscInt      dim, id = 1;
  void         *ctx;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
  PetscCall(PetscDSGetSpatialDimension(ds, &dim));
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  switch (user->obsType) {
  case OBSTACLE_BALL:
    PetscCall(PetscDSSetResidual(ds, 0, NULL, f1_u));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
    PetscCall(PetscDSSetLowerBound(ds, 0, obstacle_ball, param));
    exact = exactSol_ball;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ds), PETSC_ERR_ARG_WRONG, "Invalid obstacle type: %s (%d)", obstacleTypes[PetscMin(user->obsType, NUM_OBSTACLE_TYPES)], user->obsType);
  }
  PetscCall(PetscBagGetData(user->bag, (void **)&ctx));
  PetscCall(PetscDSSetExactSolution(ds, 0, exact, ctx));
  PetscCall(DMGetLabel(dm, "marker", &label));
  if (label) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)exact, NULL, ctx, NULL));
  /* Setup constants */
  {
    PetscScalar constants[4];

    constants[0] = param->r_0;    // Ball radius
    constants[1] = param->r_free; // Radius of the free boundary for the ball obstacle
    constants[2] = param->A;      // Logarithmic coefficient in exact ball solution
    constants[3] = param->B;      // Constant coefficient in exact ball solution
    PetscCall(PetscDSSetConstants(ds, 4, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupFE(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), void *ctx)
{
  AppCtx        *user = (AppCtx *)ctx;
  DM             cdm  = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBegin;
  /* Create finite element */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, name));
  // Set discretization and boundary conditions for each mesh
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;   /* Problem specification */
  SNES   snes; /* Nonlinear solver */
  Vec    u;    /* Solutions */
  AppCtx user; /* User-defined work context */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &user.bag));
  PetscCall(SetupParameters(PETSC_COMM_WORLD, &user));
  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupFE(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject)u, "potential"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &user));
  PetscCall(DMPlexSetSNESVariableBounds(dm, snes));

  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetSolution(snes, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-potential_view"));
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscBagDestroy(&user.bag));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_plex_box_lower -2.,-2. -dm_plex_box_upper 2.,2. -dm_plex_box_faces 20,20 \
          -potential_petscspace_degree 1 \
          -snes_type vinewtonrsls -snes_vi_zero_tolerance 1.0e-12 \
          -ksp_type preonly -pc_type lu

    # Check the exact solution
    test:
      suffix: ball_0
      requires: triangle
      args: -dmsnes_check

    # Check convergence
    test:
      suffix: ball_1
      requires: triangle
      args: -snes_convergence_estimate -convest_num_refine 2

    # Check different size obstacle
    test:
      suffix: ball_2
      requires: triangle
      args: -r_0 0.4
      output_file: output/empty.out

    # Check quadrilateral mesh
    test:
      suffix: ball_3
      args: -dm_plex_simplex 0 -snes_convergence_estimate -convest_num_refine 2

TEST*/
