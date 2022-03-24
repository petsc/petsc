static char help[] = "Tests mesh adaptation with DMPlex and pragmatic.\n";

#include <petsc/private/dmpleximpl.h>

#include <petscsnes.h>

typedef struct {
  PetscInt  Nr;         /* The number of refinement passes */
  PetscInt  metOpt;     /* Different choices of metric */
  PetscReal hmax, hmin; /* Max and min sizes prescribed by the metric */
  PetscBool doL2;       /* Test L2 projection */
} AppCtx;

/*
Classic hyperbolic sensor function for testing multi-scale anisotropic mesh adaptation:

  f:[-1, 1]x[-1, 1] \to R,
    f(x, y) = sin(50xy)/100 if |xy| > 2\pi/50 else sin(50xy)

(mapped to have domain [0,1] x [0,1] in this case).
*/
static PetscErrorCode sensor(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  const PetscReal xref = 2.*x[0] - 1.;
  const PetscReal yref = 2.*x[1] - 1.;
  const PetscReal xy   = xref*yref;

  PetscFunctionBeginUser;
  u[0] = PetscSinReal(50.*xy);
  if (PetscAbsReal(xy) > 2.*PETSC_PI/50.) u[0] *= 0.01;
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->Nr     = 1;
  options->metOpt = 1;
  options->hmin   = 0.05;
  options->hmax   = 0.5;
  options->doL2   = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-Nr", "Numberof refinement passes", "ex19.c", options->Nr, &options->Nr, NULL, 1));
  CHKERRQ(PetscOptionsBoundedInt("-met", "Different choices of metric", "ex19.c", options->metOpt, &options->metOpt, NULL,0));
  CHKERRQ(PetscOptionsReal("-hmax", "Max size prescribed by the metric", "ex19.c", options->hmax, &options->hmax, NULL));
  CHKERRQ(PetscOptionsReal("-hmin", "Min size prescribed by the metric", "ex19.c", options->hmin, &options->hmin, NULL));
  CHKERRQ(PetscOptionsBool("-do_L2", "Test L2 projection", "ex19.c", options->doL2, &options->doL2, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(PetscObjectSetName((PetscObject) *dm, "DMinit"));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-init_dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMetricSensor(DM dm, AppCtx *user, Vec *metric)
{
  PetscSimplePointFunc funcs[1] = {sensor};
  DM             dmSensor, dmGrad, dmHess;
  PetscFE        fe;
  Vec            f, g, H;
  PetscBool      simplex;
  PetscInt       dim;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));

  CHKERRQ(DMClone(dm, &dmSensor));
  CHKERRQ(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, simplex, 1, -1, &fe));
  CHKERRQ(DMSetField(dmSensor, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dmSensor));
  CHKERRQ(DMCreateLocalVector(dmSensor, &f));
  CHKERRQ(DMProjectFunctionLocal(dmSensor, 0., funcs, NULL, INSERT_VALUES, f));
  CHKERRQ(VecViewFromOptions(f, NULL, "-sensor_view"));

  // Recover the gradient of the sensor function
  CHKERRQ(DMClone(dm, &dmGrad));
  CHKERRQ(PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, simplex, 1, -1, &fe));
  CHKERRQ(DMSetField(dmGrad, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dmGrad));
  CHKERRQ(DMCreateLocalVector(dmGrad, &g));
  CHKERRQ(DMPlexComputeGradientClementInterpolant(dmSensor, f, g));
  CHKERRQ(VecDestroy(&f));
  CHKERRQ(VecViewFromOptions(g, NULL, "-gradient_view"));

  // Recover the Hessian of the sensor function
  CHKERRQ(DMClone(dm, &dmHess));
  CHKERRQ(DMPlexMetricCreate(dmHess, 0, &H));
  CHKERRQ(DMPlexComputeGradientClementInterpolant(dmGrad, g, H));
  CHKERRQ(VecDestroy(&g));
  CHKERRQ(VecViewFromOptions(H, NULL, "-hessian_view"));

  // Obtain a metric by Lp normalization
  CHKERRQ(DMPlexMetricNormalize(dmHess, H, PETSC_TRUE, PETSC_TRUE, metric));
  CHKERRQ(VecDestroy(&H));
  CHKERRQ(DMDestroy(&dmHess));
  CHKERRQ(DMDestroy(&dmGrad));
  CHKERRQ(DMDestroy(&dmSensor));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMetric(DM dm, AppCtx *user, Vec *metric)
{
  PetscReal          lambda = 1/(user->hmax*user->hmax);

  PetscFunctionBeginUser;
  if (user->metOpt == 0) {
    /* Specify a uniform, isotropic metric */
    CHKERRQ(DMPlexMetricCreateUniform(dm, 0, lambda, metric));
  } else if (user->metOpt == 3) {
    CHKERRQ(ComputeMetricSensor(dm, user, metric));
  } else {
    DM                 cdm;
    Vec                coordinates;
    const PetscScalar *coords;
    PetscScalar       *met;
    PetscReal          h;
    PetscInt           dim, i, j, vStart, vEnd, v;

    CHKERRQ(DMPlexMetricCreate(dm, 0, metric));
    CHKERRQ(DMGetDimension(dm, &dim));
    CHKERRQ(DMGetCoordinateDM(dm, &cdm));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(VecGetArrayRead(coordinates, &coords));
    CHKERRQ(VecGetArray(*metric, &met));
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    for (v = vStart; v < vEnd; ++v) {
      PetscScalar *vcoords;
      PetscScalar *pmet;

      CHKERRQ(DMPlexPointLocalRead(cdm, v, coords, &vcoords));
      switch (user->metOpt) {
      case 1:
        h = user->hmax - (user->hmax-user->hmin)*PetscRealPart(vcoords[0]);
        break;
      case 2:
        h = user->hmax*PetscAbsReal(((PetscReal) 1.0)-PetscExpReal(-PetscAbsScalar(vcoords[0]-(PetscReal)0.5))) + user->hmin;
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "metOpt = 0, 1, 2 or 3, cannot be %d", user->metOpt);
      }
      CHKERRQ(DMPlexPointLocalRef(dm, v, met, &pmet));
      for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
          if (i == j) {
            if (i == 0) pmet[i*dim+j] = 1/(h*h);
            else pmet[i*dim+j] = lambda;
          } else pmet[i*dim+j] = 0.0;
        }
      }
    }
    CHKERRQ(VecRestoreArray(*metric, &met));
    CHKERRQ(VecRestoreArrayRead(coordinates, &coords));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0] + x[1];
  return 0;
}

static PetscErrorCode TestL2Projection(DM dm, DM dma, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *) = {linear};
  DM               dmProj, dmaProj;
  PetscFE          fe;
  KSP              ksp;
  Mat              Interp, mass, mass2;
  Vec              u, ua, scaling, rhs, uproj;
  PetscReal        error;
  PetscBool        simplex;
  PetscInt         dim;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));

  CHKERRQ(DMClone(dm, &dmProj));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(DMSetField(dmProj, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dmProj));

  CHKERRQ(DMClone(dma, &dmaProj));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(DMSetField(dmaProj, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dmaProj));

  CHKERRQ(DMGetGlobalVector(dmProj, &u));
  CHKERRQ(DMGetGlobalVector(dmaProj, &ua));
  CHKERRQ(DMGetGlobalVector(dmaProj, &rhs));
  CHKERRQ(DMGetGlobalVector(dmaProj, &uproj));

  // Interpolate onto original mesh using dual basis
  CHKERRQ(DMProjectFunction(dmProj, 0.0, funcs, NULL, INSERT_VALUES, u));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "Original"));
  CHKERRQ(VecViewFromOptions(u, NULL, "-orig_vec_view"));
  CHKERRQ(DMComputeL2Diff(dmProj, 0.0, funcs, NULL, u, &error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Original L2 Error: %g\n", (double) error));
  // Interpolate onto NEW mesh using dual basis
  CHKERRQ(DMProjectFunction(dmaProj, 0.0, funcs, NULL, INSERT_VALUES, ua));
  CHKERRQ(PetscObjectSetName((PetscObject) ua, "Adapted"));
  CHKERRQ(VecViewFromOptions(ua, NULL, "-adapt_vec_view"));
  CHKERRQ(DMComputeL2Diff(dmaProj, 0.0, funcs, NULL, ua, &error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Adapted L2 Error: %g\n", (double) error));
  // Interpolate between meshes using interpolation matrix
  CHKERRQ(DMCreateInterpolation(dmProj, dmaProj, &Interp, &scaling));
  CHKERRQ(MatInterpolate(Interp, u, ua));
  CHKERRQ(MatDestroy(&Interp));
  CHKERRQ(VecDestroy(&scaling));
  CHKERRQ(PetscObjectSetName((PetscObject) ua, "Interpolation"));
  CHKERRQ(VecViewFromOptions(ua, NULL, "-interp_vec_view"));
  CHKERRQ(DMComputeL2Diff(dmaProj, 0.0, funcs, NULL, ua, &error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Interpolated L2 Error: %g\n", (double) error));
  // L2 projection
  CHKERRQ(DMCreateMassMatrix(dmaProj, dmaProj, &mass));
  CHKERRQ(MatViewFromOptions(mass, NULL, "-mass_mat_view"));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, mass, mass));
  CHKERRQ(KSPSetFromOptions(ksp));
  //   Compute rhs as M f, could also direclty project the analytic function but we might not have it
  CHKERRQ(DMCreateMassMatrix(dmProj, dmaProj, &mass2));
  CHKERRQ(MatMult(mass2, u, rhs));
  CHKERRQ(MatDestroy(&mass2));
  CHKERRQ(KSPSolve(ksp, rhs, uproj));
  CHKERRQ(PetscObjectSetName((PetscObject) uproj, "L_2 Projection"));
  CHKERRQ(VecViewFromOptions(uproj, NULL, "-proj_vec_view"));
  CHKERRQ(DMComputeL2Diff(dmaProj, 0.0, funcs, NULL, uproj, &error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Projected L2 Error: %g\n", (double) error));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&mass));
  CHKERRQ(DMRestoreGlobalVector(dmProj, &u));
  CHKERRQ(DMRestoreGlobalVector(dmaProj, &ua));
  CHKERRQ(DMRestoreGlobalVector(dmaProj, &rhs));
  CHKERRQ(DMRestoreGlobalVector(dmaProj, &uproj));
  CHKERRQ(DMDestroy(&dmProj));
  CHKERRQ(DMDestroy(&dmaProj));
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  MPI_Comm       comm;
  DM             dma, odm;
  Vec            metric;
  PetscInt       r;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(ProcessOptions(comm, &user));
  CHKERRQ(CreateMesh(comm, &dm));

  odm  = dm;
  CHKERRQ(DMPlexDistributeOverlap(odm, 1, NULL, &dm));
  if (!dm) {dm = odm;}
  else     CHKERRQ(DMDestroy(&odm));

  for (r = 0; r < user.Nr; ++r) {
    DMLabel label;

    CHKERRQ(ComputeMetric(dm, &user, &metric));
    CHKERRQ(DMGetLabel(dm, "marker", &label));
    CHKERRQ(DMAdaptMetric(dm, metric, label, NULL, &dma));
    CHKERRQ(VecDestroy(&metric));
    CHKERRQ(PetscObjectSetName((PetscObject) dma, "DMadapt"));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) dma, "adapt_"));
    CHKERRQ(DMViewFromOptions(dma, NULL, "-dm_view"));
    if (user.doL2) CHKERRQ(TestL2Projection(dm, dma, &user));
    CHKERRQ(DMDestroy(&dm));
    dm   = dma;
  }
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) dm, "final_"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: pragmatic

  testset:
    args: -dm_plex_box_faces 4,4,4 -dm_adaptor pragmatic -met 2 -init_dm_view -adapt_dm_view -dm_adaptor pragmatic

    test:
      suffix: 0
      args: -dm_plex_separate_marker 0
    test:
      suffix: 1
      args: -dm_plex_separate_marker 1
    test:
      suffix: 2
      args: -dm_plex_dim 3
    test:
      suffix: 3
      args: -dm_plex_dim 3

  # Pragmatic hangs for simple partitioner
  testset:
    requires: parmetis
    args: -dm_plex_box_faces 2,2 -dm_adaptor pragmatic -petscpartitioner_type parmetis -met 2 -init_dm_view -adapt_dm_view -dm_adaptor pragmatic

    test:
      suffix: 4
      nsize: 2
    test:
      suffix: 5
      nsize: 4

  test:
    requires: parmetis
    suffix: 6
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_box_faces 9,9,9 -dm_adaptor pragmatic -petscpartitioner_type parmetis \
          -met 0 -hmin 0.01 -hmax 0.03 -init_dm_view -adapt_dm_view -dm_adaptor pragmatic
  test:
    requires: parmetis
    suffix: 7
    nsize: 2
    args: -dm_plex_box_faces 19,19 -dm_adaptor pragmatic -petscpartitioner_type parmetis \
          -met 2 -hmax 0.5 -hmin 0.001 -init_dm_view -adapt_dm_view -dm_adaptor pragmatic
  test:
    suffix: proj_0
    args: -dm_plex_box_faces 2,2 -dm_plex_hash_location -dm_adaptor pragmatic -init_dm_view -adapt_dm_view -do_L2 \
          -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu -dm_adaptor pragmatic
  test:
    suffix: proj_1
    args: -dm_plex_box_faces 4,4 -dm_plex_hash_location -dm_adaptor pragmatic -init_dm_view -adapt_dm_view -do_L2 \
          -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu -dm_adaptor pragmatic

  test:
    suffix: sensor
    args: -dm_plex_box_faces 9,9 -met 3 -dm_adaptor pragmatic -init_dm_view -adapt_dm_view \
          -dm_plex_metric_h_min 1.e-10 -dm_plex_metric_h_max 1.0e-01 -dm_plex_metric_a_max 1.0e+05 -dm_plex_metric_p 1.0 \
            -dm_plex_metric_target_complexity 10000.0 -dm_adaptor pragmatic

TEST*/
