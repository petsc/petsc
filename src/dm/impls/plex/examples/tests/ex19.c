static char help[] = "Tests mesh adaptation with DMPlex and pragmatic.\n";

#include <petsc/private/dmpleximpl.h>

#include <petscksp.h>

typedef struct {
  DM        dm;
  /* Definition of the test case (mesh and metric field) */
  PetscInt  dim;                         /* The topological mesh dimension */
  char      mshNam[PETSC_MAX_PATH_LEN];  /* Name of the mesh filename if any */
  PetscInt  nbrVerEdge;                  /* Number of vertices per edge if unit square/cube generated */
  char      bdLabel[PETSC_MAX_PATH_LEN]; /* Name of the label marking boundary facets */
  PetscInt  metOpt;                      /* Different choices of metric */
  PetscReal hmax, hmin;                  /* Max and min sizes prescribed by the metric */
  PetscBool doL2;                        /* Test L2 projection */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim        = 2;
  ierr = PetscStrcpy(options->mshNam, "");CHKERRQ(ierr);
  options->nbrVerEdge = 5;
  ierr = PetscStrcpy(options->bdLabel, "");CHKERRQ(ierr);
  options->metOpt     = 1;
  options->hmin       = 0.05;
  options->hmax       = 0.5;
  options->doL2       = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex19.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsString("-msh", "Name of the mesh filename if any", "ex19.c", options->mshNam, options->mshNam, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-nbrVerEdge", "Number of vertices per edge if unit square/cube generated", "ex19.c", options->nbrVerEdge, &options->nbrVerEdge, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsString("-bdLabel", "Name of the label marking boundary facets", "ex19.c", options->bdLabel, options->bdLabel, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-met", "Different choices of metric", "ex19.c", options->metOpt, &options->metOpt, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmax", "Max size prescribed by the metric", "ex19.c", options->hmax, &options->hmax, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmin", "Min size prescribed by the metric", "ex19.c", options->hmin, &options->hmin, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-do_L2", "Test L2 projection", "ex19.c", options->doL2, &options->doL2, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user)
{
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcmp(user->mshNam, "", &flag);CHKERRQ(ierr);
  if (flag) {
    PetscInt faces[3];
    faces[0] = user->nbrVerEdge-1;
    faces[1] = user->nbrVerEdge-1;
    faces[2] = user->nbrVerEdge-1;

    ierr = DMPlexCreateBoxMesh(comm, user->dim, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->mshNam, PETSC_TRUE, &user->dm);CHKERRQ(ierr);
    ierr = DMGetDimension(user->dm, &user->dim);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(user->dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(&user->dm);CHKERRQ(ierr);
      user->dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(user->dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMetric(DM dm, AppCtx *user, Vec *metric)
{
  DM                 cdm, mdm;
  PetscSection       csec, msec;
  Vec                coordinates;
  const PetscScalar *coords;
  PetscScalar       *met;
  PetscReal          h, *lambda, lbd, lmax;
  PetscInt           pStart, pEnd, p, d;
  const PetscInt     dim = user->dim, Nd = dim*dim;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = PetscCalloc1(PetscMax(3, dim),&lambda);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMClone(cdm, &mdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &csec);CHKERRQ(ierr);

  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &msec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(msec, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(msec, 0, Nd);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(csec, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(msec, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    ierr = PetscSectionSetDof(msec, p, Nd);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(msec, p, 0, Nd);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(msec);CHKERRQ(ierr);
  ierr = DMSetLocalSection(mdm, msec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&msec);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(mdm, metric);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(*metric, &met);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscScalar       *pcoords;
    PetscScalar       *pmet;

    ierr = DMPlexPointLocalRead(cdm, p, coords, &pcoords);CHKERRQ(ierr);
    switch (user->metOpt) {
    case 0:
      lbd = 1/(user->hmax*user->hmax);
      lambda[0] = lambda[1] = lambda[2] = lbd;
      break;
    case 1:
      h = user->hmax - (user->hmax-user->hmin)*PetscRealPart(pcoords[0]);
      h = h*h;
      lmax = 1/(user->hmax*user->hmax);
      lambda[0] = 1/h;
      lambda[1] = lmax;
      lambda[2] = lmax;
      break;
    case 2:
      h = user->hmax*PetscAbsReal(((PetscReal) 1.0)-PetscExpReal(-PetscAbsScalar(pcoords[0]-(PetscReal)0.5))) + user->hmin;
      lbd = 1/(h*h);
      lmax = 1/(user->hmax*user->hmax);
      lambda[0] = lbd;
      lambda[1] = lmax;
      lambda[2] = lmax;
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "metOpt = 0, 1 or 2, cannot be %d", user->metOpt);
    }
    /* Only set the diagonal */
    ierr = DMPlexPointLocalRef(mdm, p, met, &pmet);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) pmet[d*(dim+1)] = lambda[d];
  }
  ierr = VecRestoreArray(*metric, &met);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMDestroy(&mdm);CHKERRQ(ierr);
  ierr = PetscFree(lambda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0] + x[1];
  return 0;
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode TestL2Projection(DM dm, DM dma, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  KSP              ksp;
  PetscDS          prob;
  PetscFE          fe;
  Mat              Interp, mass;
  Vec              u, ua, scaling, ones, massLumped, rhs, uproj;
  PetscReal        error;
  PetscInt         dim;
  MPI_Comm         comm;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dma, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);

  funcs[0] = linear;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &ua);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &ones);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &massLumped);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &rhs);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &uproj);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Original");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-orig_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, 0.0, funcs, NULL, u, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Original L2 Error: %g\n", (double) error);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dm, dma, &Interp, &scaling);CHKERRQ(ierr);
  ierr = MatInterpolate(Interp, u, ua);CHKERRQ(ierr);
  ierr = MatDestroy(&Interp);CHKERRQ(ierr);
  ierr = VecDestroy(&scaling);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) ua, "Interpolation");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ua, NULL, "-interp_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dma, 0.0, funcs, NULL, ua, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Interpolated L2 Error: %g\n", (double) error);CHKERRQ(ierr);

  ierr = VecSet(ones, 1.0);CHKERRQ(ierr);
  ierr = DMPlexComputeJacobianAction(dma, NULL, 0, 0, ua, NULL, ones, massLumped, user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) massLumped, "Lumped mass");CHKERRQ(ierr);
  ierr = VecViewFromOptions(massLumped, NULL, "-mass_vec_view");CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(dm, dma, &mass);CHKERRQ(ierr);
  ierr = MatMult(mass, u, rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);
  ierr = VecViewFromOptions(rhs, NULL, "-lumped_rhs_view");CHKERRQ(ierr);
  ierr = VecPointwiseDivide(uproj, rhs, massLumped);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uproj, "Different Lumped Projection");CHKERRQ(ierr);
  ierr = VecViewFromOptions(uproj, NULL, "-lumped_rhs_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dma, 0.0, funcs, NULL, uproj, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Lumped (rhs) L2 Error: %g\n", (double) error);CHKERRQ(ierr);

  ierr = DMCreateMatrix(dma, &mass);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dma, ua, mass, mass, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(mass, NULL, "-mass_mat_view");CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mass, mass);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uproj, "Full Projection");CHKERRQ(ierr);
  ierr = VecViewFromOptions(uproj, NULL, "-proj_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dma, 0.0, funcs, NULL, uproj, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Projected L2 Error: %g\n", (double) error);CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &ua);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &ones);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &massLumped);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &rhs);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &uproj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  AppCtx         user;                 /* user-defined work context */
  DMLabel        bdLabel = NULL;
  MPI_Comm       comm;
  DM             dma, odm;
  Vec            metric;
  size_t         len;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  ierr = CreateMesh(comm, &user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) user.dm, "DMinit");CHKERRQ(ierr);
  ierr = DMViewFromOptions(user.dm, NULL, "-init_dm_view");CHKERRQ(ierr);

  odm  = user.dm;
  ierr = DMPlexDistributeOverlap(odm, 1, NULL, &user.dm);CHKERRQ(ierr);
  if (!user.dm) {user.dm = odm;}
  else          {ierr = DMDestroy(&odm);CHKERRQ(ierr);}
  ierr = ComputeMetric(user.dm, &user, &metric);CHKERRQ(ierr);
  ierr = PetscStrlen(user.bdLabel, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMCreateLabel(user.dm, user.bdLabel);CHKERRQ(ierr);
    ierr = DMGetLabel(user.dm, user.bdLabel, &bdLabel);CHKERRQ(ierr);
  }
  ierr = DMAdaptMetric(user.dm, metric, bdLabel, &dma);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dma, "DMadapt");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dma, "adapt_");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dma, NULL, "-dm_view");CHKERRQ(ierr);
  if (user.doL2) {ierr = TestL2Projection(user.dm, dma, &user);CHKERRQ(ierr);}
  ierr = DMDestroy(&dma);CHKERRQ(ierr);
  ierr = VecDestroy(&metric);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: pragmatic
    args: -dim 2 -nbrVerEdge 5 -dm_plex_separate_marker 0 -met 2 -init_dm_view -adapt_dm_view
  test:
    suffix: 1
    requires: pragmatic
    args: -dim 2 -nbrVerEdge 5 -dm_plex_separate_marker 1 -bdLabel marker -met 2 -init_dm_view -adapt_dm_view
  test:
    suffix: 2
    requires: pragmatic
    args: -dim 3 -nbrVerEdge 5 -met 2 -init_dm_view -adapt_dm_view
  test:
    suffix: 3
    requires: pragmatic
    args: -dim 3 -nbrVerEdge 5 -bdLabel marker -met 2 -init_dm_view -adapt_dm_view
  test:
    suffix: 4
    requires: pragmatic
    nsize: 2
    args: -dim 2 -nbrVerEdge 3 -dm_plex_separate_marker 0 -met 2 -init_dm_view -adapt_dm_view
  test:
    suffix: 5
    requires: pragmatic
    nsize: 4
    args: -dim 2 -nbrVerEdge 3 -dm_plex_separate_marker 0 -met 2 -init_dm_view -adapt_dm_view
  test:
    suffix: 6
    requires: pragmatic
    nsize: 2
    args: -dim 3 -nbrVerEdge 10 -dm_plex_separate_marker 0 -met 0 -hmin 0.01 -hmax 0.03 -init_dm_view -adapt_dm_view
  test:
    suffix: 7
    requires: pragmatic
    nsize: 5
    args: -dim 2 -nbrVerEdge 20 -dm_plex_separate_marker 0 -met 2 -hmax 0.5 -hmin 0.001 -init_dm_view -adapt_dm_view
  test:
    suffix: proj_0
    requires: pragmatic
    args: -dim 2 -nbrVerEdge 3 -dm_plex_separate_marker 0 -init_dm_view -adapt_dm_view -do_L2 -petscspace_degree 1 -petscfe_default_quadrature_order 1 -dm_plex_hash_location -pc_type lu
  test:
    suffix: proj_1
    requires: pragmatic
    args: -dim 2 -nbrVerEdge 5 -dm_plex_separate_marker 0 -init_dm_view -adapt_dm_view -do_L2 -petscspace_degree 2 -petscfe_default_quadrature_order 4 -dm_plex_hash_location -pc_type lu

TEST*/
