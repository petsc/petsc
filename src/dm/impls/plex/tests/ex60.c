static char help[] = "Test metric utils in the uniform, isotropic case.\n\n";

#include <petscdmplex.h>

static PetscErrorCode bowl(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  *u = 0.0;
  for (d = 0; d < dim; d++) *u += 0.5*(x[d] - 0.5)*(x[d] - 0.5);

  return 0;
}

static PetscErrorCode CreateIndicator(DM dm, Vec *indicator, DM *dmIndi)
{
  MPI_Comm       comm;
  PetscFE        fe;
  PetscInt       dim;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm, &comm));
  CHKERRQ(DMClone(dm, dmIndi));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(PetscFECreateLagrange(comm, dim, 1, PETSC_TRUE, 1, PETSC_DETERMINE, &fe));
  CHKERRQ(DMSetField(*dmIndi, 0, NULL, (PetscObject)fe));
  CHKERRQ(DMCreateDS(*dmIndi));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateLocalVector(*dmIndi, indicator));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  DM              dm, dmAdapt;
  DMLabel         bdLabel = NULL, rgLabel = NULL;
  MPI_Comm        comm;
  PetscBool       uniform = PETSC_FALSE, isotropic = PETSC_FALSE, noTagging = PETSC_FALSE;
  PetscErrorCode  ierr;
  PetscInt        dim;
  PetscReal       scaling = 1.0;
  Vec             metric;

  /* Set up */
  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Mesh adaptation options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-noTagging", "Should tag preservation testing be turned off?", "ex60.c", noTagging, &noTagging, NULL));
  ierr = PetscOptionsEnd();

  /* Create box mesh */
  CHKERRQ(DMCreate(comm, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "DM_init"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-initial_mesh_view"));
  CHKERRQ(DMGetDimension(dm, &dim));

  /* Set tags to be preserved */
  if (!noTagging) {
    DM                 cdm;
    PetscInt           cStart, cEnd, c, fStart, fEnd, f, vStart, vEnd;
    const PetscScalar *coords;
    Vec                coordinates;

    /* Cell tags */
    CHKERRQ(DMCreateLabel(dm, "Cell Sets"));
    CHKERRQ(DMGetLabel(dm, "Cell Sets", &rgLabel));
    CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      PetscReal centroid[3], volume, x;

      CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL));
      x = centroid[0];
      if (x < 0.5) CHKERRQ(DMLabelSetValue(rgLabel, c, 3));
      else         CHKERRQ(DMLabelSetValue(rgLabel, c, 4));
    }

    /* Face tags */
    CHKERRQ(DMCreateLabel(dm, "Face Sets"));
    CHKERRQ(DMGetLabel(dm, "Face Sets", &bdLabel));
    CHKERRQ(DMPlexMarkBoundaryFaces(dm, 1, bdLabel));
    CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    CHKERRQ(DMGetCoordinateDM(dm, &cdm));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(VecGetArrayRead(coordinates, &coords));
    for (f = fStart; f < fEnd; ++f) {
      PetscBool flg = PETSC_TRUE;
      PetscInt *closure = NULL, closureSize, cl;
      PetscReal eps = 1.0e-08;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
      for (cl = 0; cl < closureSize*2; cl += 2) {
        PetscInt   off = closure[cl];
        PetscReal *x;

        if ((off < vStart) || (off >= vEnd)) continue;
        CHKERRQ(DMPlexPointLocalRead(cdm, off, coords, &x));
        if ((x[0] < 0.5 - eps) || (x[0] > 0.5 + eps)) flg = PETSC_FALSE;
      }
      if (flg) CHKERRQ(DMLabelSetValue(bdLabel, f, 2));
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    }
    CHKERRQ(VecRestoreArrayRead(coordinates, &coords));
  }

  /* Construct metric */
  CHKERRQ(DMPlexMetricSetFromOptions(dm));
  CHKERRQ(DMPlexMetricIsUniform(dm, &uniform));
  CHKERRQ(DMPlexMetricIsIsotropic(dm, &isotropic));
  if (uniform) {
    CHKERRQ(DMPlexMetricCreateUniform(dm, 0, scaling, &metric));
  }
  else {
    DM  dmIndi;
    Vec indicator;

    /* Construct "error indicator" */
    CHKERRQ(CreateIndicator(dm, &indicator, &dmIndi));
    if (isotropic) {

      /* Isotropic case: just specify unity */
      CHKERRQ(VecSet(indicator, scaling));
      CHKERRQ(DMPlexMetricCreateIsotropic(dm, 0, indicator, &metric));

    } else {
      PetscFE fe;

      /* 'Anisotropic' case: approximate the identity by recovering the Hessian of a parabola */
      DM               dmGrad;
      PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*) = {bowl};
      Vec              gradient;

      /* Project the parabola into P1 space */
      CHKERRQ(DMProjectFunctionLocal(dmIndi, 0.0, funcs, NULL, INSERT_ALL_VALUES, indicator));

      /* Approximate the gradient */
      CHKERRQ(DMClone(dmIndi, &dmGrad));
      CHKERRQ(PetscFECreateLagrange(comm, dim, dim, PETSC_TRUE, 1, PETSC_DETERMINE, &fe));
      CHKERRQ(DMSetField(dmGrad, 0, NULL, (PetscObject)fe));
      CHKERRQ(DMCreateDS(dmGrad));
      CHKERRQ(PetscFEDestroy(&fe));
      CHKERRQ(DMCreateLocalVector(dmGrad, &gradient));
      CHKERRQ(DMPlexComputeGradientClementInterpolant(dmIndi, indicator, gradient));
      CHKERRQ(VecViewFromOptions(gradient, NULL, "-adapt_gradient_view"));

      /* Approximate the Hessian */
      CHKERRQ(DMPlexMetricCreate(dm, 0, &metric));
      CHKERRQ(DMPlexComputeGradientClementInterpolant(dmGrad, gradient, metric));
      CHKERRQ(VecViewFromOptions(metric, NULL, "-adapt_hessian_view"));
      CHKERRQ(VecDestroy(&gradient));
      CHKERRQ(DMDestroy(&dmGrad));
    }
    CHKERRQ(VecDestroy(&indicator));
    CHKERRQ(DMDestroy(&dmIndi));
  }

  /* Test metric routines */
  {
    DM        dmDet;
    PetscReal errornorm, norm, tol = 1.0e-10, weights[2] = {0.8, 0.2};
    Vec       metric1, metric2, metricComb, determinant;
    Vec       metrics[2];

    CHKERRQ(VecDuplicate(metric, &metric1));
    CHKERRQ(VecSet(metric1, 0));
    CHKERRQ(VecAXPY(metric1, 0.625, metric));
    CHKERRQ(VecDuplicate(metric, &metric2));
    CHKERRQ(VecSet(metric2, 0));
    CHKERRQ(VecAXPY(metric2, 2.5, metric));
    metrics[0] = metric1;
    metrics[1] = metric2;

    /* Test metric average */
    CHKERRQ(DMPlexMetricAverage(dm, 2, weights, metrics, &metricComb));
    CHKERRQ(VecAXPY(metricComb, -1, metric));
    CHKERRQ(VecNorm(metric, NORM_2, &norm));
    CHKERRQ(VecNorm(metricComb, NORM_2, &errornorm));
    errornorm /= norm;
    CHKERRQ(PetscPrintf(comm, "Metric average L2 error: %.4f%%\n", 100*errornorm));
    PetscCheckFalse(errornorm > tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric average test failed");
    CHKERRQ(VecDestroy(&metricComb));

    /* Test metric intersection */
    if (isotropic) {
      CHKERRQ(DMPlexMetricIntersection(dm, 2, metrics, &metricComb));
      CHKERRQ(VecAXPY(metricComb, -1, metric1));
      CHKERRQ(VecNorm(metricComb, NORM_2, &errornorm));
      errornorm /= norm;
      CHKERRQ(PetscPrintf(comm, "Metric intersection L2 error: %.4f%%\n", 100*errornorm));
      PetscCheckFalse(errornorm > tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric intersection test failed");
    }
    CHKERRQ(VecDestroy(&metric1));
    CHKERRQ(VecDestroy(&metric2));
    CHKERRQ(VecDestroy(&metricComb));

    /* Test metric SPD enforcement */
    CHKERRQ(DMPlexMetricEnforceSPD(dm, metric, PETSC_TRUE, PETSC_TRUE, &metric1, &determinant));
    if (isotropic) {
      Vec err;

      CHKERRQ(VecDuplicate(determinant, &err));
      CHKERRQ(VecSet(err, 1.0));
      CHKERRQ(VecNorm(err, NORM_2, &norm));
      CHKERRQ(VecAXPY(err, -1, determinant));
      CHKERRQ(VecNorm(err, NORM_2, &errornorm));
      CHKERRQ(VecDestroy(&err));
      errornorm /= norm;
      CHKERRQ(PetscPrintf(comm, "Metric determinant L2 error: %.4f%%\n", 100*errornorm));
      PetscCheckFalse(errornorm > tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Determinant is not unit");
      CHKERRQ(VecAXPY(metric1, -1, metric));
      CHKERRQ(VecNorm(metric1, NORM_2, &errornorm));
      errornorm /= norm;
      CHKERRQ(PetscPrintf(comm, "Metric SPD enforcement L2 error: %.4f%%\n", 100*errornorm));
      PetscCheckFalse(errornorm > tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric SPD enforcement test failed");
    }
    CHKERRQ(VecDestroy(&metric1));
    CHKERRQ(VecGetDM(determinant, &dmDet));
    CHKERRQ(VecDestroy(&determinant));
    CHKERRQ(DMDestroy(&dmDet));

    /* Test metric normalization */
    CHKERRQ(DMPlexMetricNormalize(dm, metric, PETSC_TRUE, PETSC_TRUE, &metric1));
    if (isotropic) {
      PetscReal target;

      CHKERRQ(DMPlexMetricGetTargetComplexity(dm, &target));
      scaling = PetscPowReal(target, 2.0/dim);
      if (uniform) {
        CHKERRQ(DMPlexMetricCreateUniform(dm, 0, scaling, &metric2));
      } else {
        DM  dmIndi;
        Vec indicator;

        CHKERRQ(CreateIndicator(dm, &indicator, &dmIndi));
        CHKERRQ(VecSet(indicator, scaling));
        CHKERRQ(DMPlexMetricCreateIsotropic(dm, 0, indicator, &metric2));
        CHKERRQ(DMDestroy(&dmIndi));
        CHKERRQ(VecDestroy(&indicator));
      }
      CHKERRQ(VecAXPY(metric2, -1, metric1));
      CHKERRQ(VecNorm(metric2, NORM_2, &errornorm));
      errornorm /= norm;
      CHKERRQ(PetscPrintf(comm, "Metric normalization L2 error: %.4f%%\n", 100*errornorm));
      PetscCheckFalse(errornorm > tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric normalization test failed");
    }
    CHKERRQ(VecCopy(metric1, metric));
    CHKERRQ(VecDestroy(&metric2));
    CHKERRQ(VecDestroy(&metric1));
  }

  /* Adapt the mesh */
  CHKERRQ(DMAdaptMetric(dm, metric, bdLabel, rgLabel, &dmAdapt));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dmAdapt, "DM_adapted"));
  CHKERRQ(VecDestroy(&metric));
  CHKERRQ(DMViewFromOptions(dmAdapt, NULL, "-adapted_mesh_view"));

  /* Test tag preservation */
  if (!noTagging) {
    PetscBool hasTag;
    PetscInt  size;

    CHKERRQ(DMGetLabel(dmAdapt, "Face Sets", &bdLabel));
    CHKERRQ(DMLabelHasStratum(bdLabel, 1, &hasTag));
    PetscCheckFalse(!hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have face tag 1");
    CHKERRQ(DMLabelHasStratum(bdLabel, 2, &hasTag));
    PetscCheckFalse(!hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have face tag 2");
    CHKERRQ(DMLabelGetNumValues(bdLabel, &size));
    PetscCheckFalse(size != 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh has the wrong number of face tags (got %d, expected 2)", size);

    CHKERRQ(DMGetLabel(dmAdapt, "Cell Sets", &rgLabel));
    CHKERRQ(DMLabelHasStratum(rgLabel, 3, &hasTag));
    PetscCheckFalse(!hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have cell tag 3");
    CHKERRQ(DMLabelHasStratum(rgLabel, 4, &hasTag));
    PetscCheckFalse(!hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have cell tag 4");
    CHKERRQ(DMLabelGetNumValues(rgLabel, &size));
    PetscCheckFalse(size != 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh has the wrong number of cell tags (got %d, expected 2)", size);
  }

  /* Clean up */
  CHKERRQ(DMDestroy(&dmAdapt));
  ierr = PetscFinalize();
  return 0;
}

/*TEST

  testset:
    requires: pragmatic
    args: -dm_plex_box_faces 4,4 -dm_plex_metric_target_complexity 100 -dm_adaptor pragmatic -noTagging

    test:
      suffix: uniform_2d_pragmatic
      args: -dm_plex_metric_uniform
    test:
      suffix: iso_2d_pragmatic
      args: -dm_plex_metric_isotropic
    test:
      suffix: hessian_2d_pragmatic

  testset:
    requires: pragmatic tetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -dm_plex_metric_target_complexity 100 -dm_adaptor pragmatic -noTagging

    test:
      suffix: uniform_3d_pragmatic
      args: -dm_plex_metric_uniform -noTagging
    test:
      suffix: iso_3d_pragmatic
      args: -dm_plex_metric_isotropic -noTagging
    test:
      suffix: hessian_3d_pragmatic

  testset:
    requires: mmg
    args: -dm_plex_box_faces 4,4 -dm_plex_metric_target_complexity 100 -dm_adaptor mmg

    test:
      suffix: uniform_2d_mmg
      args: -dm_plex_metric_uniform
    test:
      suffix: iso_2d_mmg
      args: -dm_plex_metric_isotropic
    test:
      suffix: hessian_2d_mmg

  testset:
    requires: mmg tetgen
    args: -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -dm_plex_metric_target_complexity 100 -dm_adaptor mmg

    test:
      suffix: uniform_3d_mmg
      args: -dm_plex_metric_uniform
    test:
      suffix: iso_3d_mmg
      args: -dm_plex_metric_isotropic
    test:
      suffix: hessian_3d_mmg

  testset:
    requires: parmmg tetgen
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -dm_plex_metric_target_complexity 100 -dm_adaptor parmmg

    test:
      suffix: uniform_3d_parmmg
      args: -dm_plex_metric_uniform
    test:
      suffix: iso_3d_parmmg
      args: -dm_plex_metric_isotropic
    test:
      suffix: hessian_3d_parmmg

TEST*/
