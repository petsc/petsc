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
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMClone(dm, dmIndi));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFECreateLagrange(comm, dim, 1, PETSC_TRUE, 1, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(*dmIndi, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(*dmIndi));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateLocalVector(*dmIndi, indicator));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  DM              dm, dmAdapt;
  DMLabel         bdLabel = NULL, rgLabel = NULL;
  MPI_Comm        comm;
  PetscBool       uniform = PETSC_FALSE, isotropic = PETSC_FALSE, noTagging = PETSC_FALSE;
  PetscInt        dim;
  PetscReal       scaling = 1.0;
  Vec             metric;

  /* Set up */
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, "", "Mesh adaptation options", "DMPLEX");
  PetscCall(PetscOptionsBool("-noTagging", "Should tag preservation testing be turned off?", "ex60.c", noTagging, &noTagging, NULL));
  PetscOptionsEnd();

  /* Create box mesh */
  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject) dm, "DM_init"));
  PetscCall(DMViewFromOptions(dm, NULL, "-initial_mesh_view"));
  PetscCall(DMGetDimension(dm, &dim));

  /* Set tags to be preserved */
  if (!noTagging) {
    DM                 cdm;
    PetscInt           cStart, cEnd, c, fStart, fEnd, f, vStart, vEnd;
    const PetscScalar *coords;
    Vec                coordinates;

    /* Cell tags */
    PetscCall(DMCreateLabel(dm, "Cell Sets"));
    PetscCall(DMGetLabel(dm, "Cell Sets", &rgLabel));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (c = cStart; c < cEnd; ++c) {
      PetscReal centroid[3], volume, x;

      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL));
      x = centroid[0];
      if (x < 0.5) PetscCall(DMLabelSetValue(rgLabel, c, 3));
      else         PetscCall(DMLabelSetValue(rgLabel, c, 4));
    }

    /* Face tags */
    PetscCall(DMCreateLabel(dm, "Face Sets"));
    PetscCall(DMGetLabel(dm, "Face Sets", &bdLabel));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, bdLabel));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(VecGetArrayRead(coordinates, &coords));
    for (f = fStart; f < fEnd; ++f) {
      PetscBool flg = PETSC_TRUE;
      PetscInt *closure = NULL, closureSize, cl;
      PetscReal eps = 1.0e-08;

      PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
      for (cl = 0; cl < closureSize*2; cl += 2) {
        PetscInt   off = closure[cl];
        PetscReal *x;

        if ((off < vStart) || (off >= vEnd)) continue;
        PetscCall(DMPlexPointLocalRead(cdm, off, coords, &x));
        if ((x[0] < 0.5 - eps) || (x[0] > 0.5 + eps)) flg = PETSC_FALSE;
      }
      if (flg) PetscCall(DMLabelSetValue(bdLabel, f, 2));
      PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    }
    PetscCall(VecRestoreArrayRead(coordinates, &coords));
  }

  /* Construct metric */
  PetscCall(DMPlexMetricSetFromOptions(dm));
  PetscCall(DMPlexMetricIsUniform(dm, &uniform));
  PetscCall(DMPlexMetricIsIsotropic(dm, &isotropic));
  if (uniform) {
    PetscCall(DMPlexMetricCreateUniform(dm, 0, scaling, &metric));
  }
  else {
    DM  dmIndi;
    Vec indicator;

    /* Construct "error indicator" */
    PetscCall(CreateIndicator(dm, &indicator, &dmIndi));
    if (isotropic) {

      /* Isotropic case: just specify unity */
      PetscCall(VecSet(indicator, scaling));
      PetscCall(DMPlexMetricCreateIsotropic(dm, 0, indicator, &metric));

    } else {
      PetscFE fe;

      /* 'Anisotropic' case: approximate the identity by recovering the Hessian of a parabola */
      DM               dmGrad;
      PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*) = {bowl};
      Vec              gradient;

      /* Project the parabola into P1 space */
      PetscCall(DMProjectFunctionLocal(dmIndi, 0.0, funcs, NULL, INSERT_ALL_VALUES, indicator));

      /* Approximate the gradient */
      PetscCall(DMClone(dmIndi, &dmGrad));
      PetscCall(PetscFECreateLagrange(comm, dim, dim, PETSC_TRUE, 1, PETSC_DETERMINE, &fe));
      PetscCall(DMSetField(dmGrad, 0, NULL, (PetscObject)fe));
      PetscCall(DMCreateDS(dmGrad));
      PetscCall(PetscFEDestroy(&fe));
      PetscCall(DMCreateLocalVector(dmGrad, &gradient));
      PetscCall(DMPlexComputeGradientClementInterpolant(dmIndi, indicator, gradient));
      PetscCall(VecViewFromOptions(gradient, NULL, "-adapt_gradient_view"));

      /* Approximate the Hessian */
      PetscCall(DMPlexMetricCreate(dm, 0, &metric));
      PetscCall(DMPlexComputeGradientClementInterpolant(dmGrad, gradient, metric));
      PetscCall(VecViewFromOptions(metric, NULL, "-adapt_hessian_view"));
      PetscCall(VecDestroy(&gradient));
      PetscCall(DMDestroy(&dmGrad));
    }
    PetscCall(VecDestroy(&indicator));
    PetscCall(DMDestroy(&dmIndi));
  }

  /* Test metric routines */
  {
    DM        dmDet;
    PetscReal errornorm, norm, tol = 1.0e-10, weights[2] = {0.8, 0.2};
    Vec       metric1, metric2, metricComb, determinant;
    Vec       metrics[2];

    PetscCall(VecDuplicate(metric, &metric1));
    PetscCall(VecSet(metric1, 0));
    PetscCall(VecAXPY(metric1, 0.625, metric));
    PetscCall(VecDuplicate(metric, &metric2));
    PetscCall(VecSet(metric2, 0));
    PetscCall(VecAXPY(metric2, 2.5, metric));
    metrics[0] = metric1;
    metrics[1] = metric2;

    /* Test metric average */
    PetscCall(DMPlexMetricAverage(dm, 2, weights, metrics, &metricComb));
    PetscCall(VecAXPY(metricComb, -1, metric));
    PetscCall(VecNorm(metric, NORM_2, &norm));
    PetscCall(VecNorm(metricComb, NORM_2, &errornorm));
    errornorm /= norm;
    PetscCall(PetscPrintf(comm, "Metric average L2 error: %.4f%%\n", (double)(100*errornorm)));
    PetscCheck(errornorm <= tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric average test failed");
    PetscCall(VecDestroy(&metricComb));

    /* Test metric intersection */
    if (isotropic) {
      PetscCall(DMPlexMetricIntersection(dm, 2, metrics, metricComb));
      PetscCall(VecAXPY(metricComb, -1, metric2));
      PetscCall(VecNorm(metricComb, NORM_2, &errornorm));
      errornorm /= norm;
      PetscCall(PetscPrintf(comm, "Metric intersection L2 error: %.4f%%\n", (double)(100*errornorm)));
      PetscCheck(errornorm <= tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric intersection test failed");
    }
    PetscCall(VecDestroy(&metric1));
    PetscCall(VecDestroy(&metric2));
    PetscCall(VecDestroy(&metricComb));

    /* Test metric SPD enforcement */
    PetscCall(DMPlexMetricEnforceSPD(dm, metric, PETSC_TRUE, PETSC_TRUE, &metric1, &determinant));
    if (isotropic) {
      Vec err;

      PetscCall(VecDuplicate(determinant, &err));
      PetscCall(VecSet(err, 1.0));
      PetscCall(VecNorm(err, NORM_2, &norm));
      PetscCall(VecAXPY(err, -1, determinant));
      PetscCall(VecNorm(err, NORM_2, &errornorm));
      PetscCall(VecDestroy(&err));
      errornorm /= norm;
      PetscCall(PetscPrintf(comm, "Metric determinant L2 error: %.4f%%\n", (double)(100*errornorm)));
      PetscCheck(errornorm <= tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Determinant is not unit");
      PetscCall(VecAXPY(metric1, -1, metric));
      PetscCall(VecNorm(metric1, NORM_2, &errornorm));
      errornorm /= norm;
      PetscCall(PetscPrintf(comm, "Metric SPD enforcement L2 error: %.4f%%\n", (double)(100*errornorm)));
      PetscCheck(errornorm <= tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric SPD enforcement test failed");
    }
    PetscCall(VecDestroy(&metric1));
    PetscCall(VecGetDM(determinant, &dmDet));
    PetscCall(VecDestroy(&determinant));
    PetscCall(DMDestroy(&dmDet));

    /* Test metric normalization */
    PetscCall(DMPlexMetricNormalize(dm, metric, PETSC_TRUE, PETSC_TRUE, &metric1));
    if (isotropic) {
      PetscReal target;

      PetscCall(DMPlexMetricGetTargetComplexity(dm, &target));
      scaling = PetscPowReal(target, 2.0/dim);
      if (uniform) {
        PetscCall(DMPlexMetricCreateUniform(dm, 0, scaling, &metric2));
      } else {
        DM  dmIndi;
        Vec indicator;

        PetscCall(CreateIndicator(dm, &indicator, &dmIndi));
        PetscCall(VecSet(indicator, scaling));
        PetscCall(DMPlexMetricCreateIsotropic(dm, 0, indicator, &metric2));
        PetscCall(DMDestroy(&dmIndi));
        PetscCall(VecDestroy(&indicator));
      }
      PetscCall(VecAXPY(metric2, -1, metric1));
      PetscCall(VecNorm(metric2, NORM_2, &errornorm));
      errornorm /= norm;
      PetscCall(PetscPrintf(comm, "Metric normalization L2 error: %.4f%%\n", (double)(100*errornorm)));
      PetscCheck(errornorm <= tol,comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric normalization test failed");
    }
    PetscCall(VecCopy(metric1, metric));
    PetscCall(VecDestroy(&metric2));
    PetscCall(VecDestroy(&metric1));
  }

  /* Adapt the mesh */
  PetscCall(DMAdaptMetric(dm, metric, bdLabel, rgLabel, &dmAdapt));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscObjectSetName((PetscObject) dmAdapt, "DM_adapted"));
  PetscCall(VecDestroy(&metric));
  PetscCall(DMViewFromOptions(dmAdapt, NULL, "-adapted_mesh_view"));

  /* Test tag preservation */
  if (!noTagging) {
    PetscBool hasTag;
    PetscInt  size;

    PetscCall(DMGetLabel(dmAdapt, "Face Sets", &bdLabel));
    PetscCall(DMLabelHasStratum(bdLabel, 1, &hasTag));
    PetscCheck(hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have face tag 1");
    PetscCall(DMLabelHasStratum(bdLabel, 2, &hasTag));
    PetscCheck(hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have face tag 2");
    PetscCall(DMLabelGetNumValues(bdLabel, &size));
    PetscCheck(size == 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh has the wrong number of face tags (got %" PetscInt_FMT ", expected 2)", size);

    PetscCall(DMGetLabel(dmAdapt, "Cell Sets", &rgLabel));
    PetscCall(DMLabelHasStratum(rgLabel, 3, &hasTag));
    PetscCheck(hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have cell tag 3");
    PetscCall(DMLabelHasStratum(rgLabel, 4, &hasTag));
    PetscCheck(hasTag,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have cell tag 4");
    PetscCall(DMLabelGetNumValues(rgLabel, &size));
    PetscCheck(size == 2,comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh has the wrong number of cell tags (got %" PetscInt_FMT ", expected 2)", size);
  }

  /* Clean up */
  PetscCall(DMDestroy(&dmAdapt));
  PetscCall(PetscFinalize());
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
