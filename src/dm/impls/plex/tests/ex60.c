static char help[] = "Test metric utils in the uniform, isotropic case.\n\n";

#include <petscdmplex.h>

static PetscErrorCode bowl(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  *u = 0.0;
  for (d = 0; d < dim; d++) *u += 0.5*(x[d] - 0.5)*(x[d] - 0.5);

  return 0;
}

int main(int argc, char **argv) {
  DM              dm, dmDist, dmAdapt;
  DMLabel         bdLabel = NULL, rgLabel = NULL;
  MPI_Comm        comm;
  PetscBool       uniform = PETSC_FALSE, isotropic = PETSC_FALSE, noTagging = PETSC_FALSE;
  PetscErrorCode  ierr;
  PetscInt       *faces, dim = 3, d;
  PetscReal       scaling = 1.0;
  Vec             metric;

  /* Set up */
  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Mesh adaptation options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex60.c", dim, &dim, NULL, 2, 3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-uniform", "Should the metric be assumed uniform?", "ex60.c", uniform, &uniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-isotropic", "Should the metric be assumed isotropic, or computed as a recovered Hessian?", "ex60.c", isotropic, &isotropic, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-noTagging", "Should tag preservation testing be turned off?", "ex60.c", noTagging, &noTagging, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  /* Create box mesh */
  ierr = PetscMalloc1(dim, &faces);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) faces[d] = 4;
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm);CHKERRQ(ierr);
  ierr = PetscFree(faces);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  /* Distribute mesh over processes */
  ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = dmDist;
  }
  ierr = PetscObjectSetName((PetscObject) dm, "DM_init");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-initial_mesh_view");CHKERRQ(ierr);

  /* Set tags to be preserved */
  if (!noTagging) {
    DM                 cdm;
    PetscInt           cStart, cEnd, c, fStart, fEnd, f, vStart, vEnd;
    const PetscScalar *coords;
    Vec                coordinates;

    /* Cell tags */
    ierr = DMCreateLabel(dm, "Cell Sets");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Cell Sets", &rgLabel);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscReal centroid[3], volume, x;

      ierr = DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL);CHKERRQ(ierr);
      x = centroid[0];
      if (x < 0.5) { ierr = DMLabelSetValue(rgLabel, c, 3);CHKERRQ(ierr); }
      else         { ierr = DMLabelSetValue(rgLabel, c, 4);CHKERRQ(ierr); }
    }

    /* Face tags */
    ierr = DMCreateLabel(dm, "Face Sets");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Face Sets", &bdLabel);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, 1, bdLabel);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      PetscBool flg = PETSC_TRUE;
      PetscInt *closure = NULL, closureSize, cl;
      PetscReal eps = 1.0e-08;

      ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        PetscInt   off = closure[cl];
        PetscReal *x;

        if ((off < vStart) || (off >= vEnd)) continue;
        ierr = DMPlexPointLocalRead(cdm, off, coords, &x);CHKERRQ(ierr);
        if ((x[0] < 0.5 - eps) || (x[0] > 0.5 + eps)) flg = PETSC_FALSE;
      }
      if (flg) { ierr = DMLabelSetValue(bdLabel, f, 2);CHKERRQ(ierr); }
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  }

  /* Construct metric */
  if (uniform) {
    if (isotropic) { ierr = DMPlexMetricCreateUniform(dm, 0, scaling, &metric);CHKERRQ(ierr); }
    else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Uniform anisotropic metrics not supported.");
  }
  else {
    DM      dmIndi;
    PetscFE fe;
    Vec     indicator;

    /* Construct "error indicator" */
    ierr = DMClone(dm, &dmIndi);CHKERRQ(ierr);
    ierr = PetscFECreateLagrange(comm, dim, 1, PETSC_TRUE, 1, PETSC_DETERMINE, &fe);CHKERRQ(ierr);
    ierr = DMSetField(dmIndi, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
    ierr = DMCreateDS(dmIndi);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dmIndi, &indicator);CHKERRQ(ierr);
    if (isotropic) {

      /* Isotropic case: just specify unity */
      ierr = VecSet(indicator, scaling);CHKERRQ(ierr);
      ierr = DMPlexMetricCreateIsotropic(dm, 0, indicator, &metric);CHKERRQ(ierr);

    } else {

      /* 'Anisotropic' case: approximate the identity by recovering the Hessian of a parabola */
      DM               dmGrad;
      PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*) = {bowl};
      Vec              gradient;

      /* Project the parabola into P1 space */
      ierr = DMProjectFunctionLocal(dmIndi, 0.0, funcs, NULL, INSERT_ALL_VALUES, indicator);CHKERRQ(ierr);

      /* Approximate the gradient */
      ierr = DMClone(dmIndi, &dmGrad);CHKERRQ(ierr);
      ierr = PetscFECreateLagrange(comm, dim, dim, PETSC_TRUE, 1, PETSC_DETERMINE, &fe);CHKERRQ(ierr);
      ierr = DMSetField(dmGrad, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
      ierr = DMCreateDS(dmGrad);CHKERRQ(ierr);
      ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(dmGrad, &gradient);CHKERRQ(ierr);
      ierr = DMPlexComputeGradientClementInterpolant(dmIndi, indicator, gradient);CHKERRQ(ierr);
      ierr = VecViewFromOptions(gradient, NULL, "-adapt_gradient_view");CHKERRQ(ierr);

      /* Approximate the Hessian */
      ierr = DMPlexMetricCreate(dm, 0, &metric);CHKERRQ(ierr);
      ierr = DMPlexComputeGradientClementInterpolant(dmGrad, gradient, metric);CHKERRQ(ierr);
      ierr = VecViewFromOptions(metric, NULL, "-adapt_hessian_view");CHKERRQ(ierr);
      ierr = VecDestroy(&gradient);CHKERRQ(ierr);
      ierr = DMDestroy(&dmGrad);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&indicator);CHKERRQ(ierr);
    ierr = DMDestroy(&dmIndi);CHKERRQ(ierr);
  }

  /* Test metric routines */
  {
    DM        dmDet;
    PetscReal errornorm, norm, tol = 1.0e-10, weights[2] = {0.8, 0.2};
    Vec       metric1, metric2, metricComb, determinant;
    Vec       metrics[2];

    ierr = VecDuplicate(metric, &metric1);CHKERRQ(ierr);
    ierr = VecSet(metric1, 0);CHKERRQ(ierr);
    ierr = VecAXPY(metric1, 0.625, metric);CHKERRQ(ierr);
    ierr = VecDuplicate(metric, &metric2);CHKERRQ(ierr);
    ierr = VecSet(metric2, 0);CHKERRQ(ierr);
    ierr = VecAXPY(metric2, 2.5, metric);CHKERRQ(ierr);
    metrics[0] = metric1;
    metrics[1] = metric2;

    /* Test metric average */
    ierr = DMPlexMetricAverage(dm, 2, weights, metrics, &metricComb);CHKERRQ(ierr);
    ierr = VecAXPY(metricComb, -1, metric);CHKERRQ(ierr);
    ierr = VecNorm(metric, NORM_2, &norm);CHKERRQ(ierr);
    ierr = VecNorm(metricComb, NORM_2, &errornorm);CHKERRQ(ierr);
    errornorm /= norm;
    ierr = PetscPrintf(comm, "Metric average L2 error: %.4f%%\n", 100*errornorm);CHKERRQ(ierr);
    if (errornorm > tol) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric average test failed");
    ierr = VecDestroy(&metricComb);CHKERRQ(ierr);

    /* Test metric intersection */
    if (isotropic) {
      ierr = DMPlexMetricIntersection(dm, 2, metrics, &metricComb);CHKERRQ(ierr);
      ierr = VecAXPY(metricComb, -1, metric1);CHKERRQ(ierr);
      ierr = VecNorm(metricComb, NORM_2, &errornorm);CHKERRQ(ierr);
      errornorm /= norm;
      ierr = PetscPrintf(comm, "Metric intersection L2 error: %.4f%%\n", 100*errornorm);CHKERRQ(ierr);
      if (errornorm > tol) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric intersection test failed");
    }
    ierr = VecDestroy(&metric1);CHKERRQ(ierr);
    ierr = VecDestroy(&metric2);CHKERRQ(ierr);
    ierr = VecDestroy(&metricComb);CHKERRQ(ierr);

    /* Test metric SPD enforcement */
    ierr = DMPlexMetricEnforceSPD(dm, metric, PETSC_TRUE, PETSC_TRUE, &metric1, &determinant);CHKERRQ(ierr);
    if (isotropic) {
      Vec err;

      ierr = VecDuplicate(determinant, &err);CHKERRQ(ierr);
      ierr = VecSet(err, 1.0);CHKERRQ(ierr);
      ierr = VecNorm(err, NORM_2, &norm);CHKERRQ(ierr);
      ierr = VecAXPY(err, -1, determinant);CHKERRQ(ierr);
      ierr = VecNorm(err, NORM_2, &errornorm);CHKERRQ(ierr);
      ierr = VecDestroy(&err);CHKERRQ(ierr);
      errornorm /= norm;
      ierr = PetscPrintf(comm, "Metric determinant L2 error: %.4f%%\n", 100*errornorm);CHKERRQ(ierr);
      if (errornorm > tol) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Determinant is not unit");
      ierr = VecAXPY(metric1, -1, metric);CHKERRQ(ierr);
      ierr = VecNorm(metric1, NORM_2, &errornorm);CHKERRQ(ierr);
      errornorm /= norm;
      ierr = PetscPrintf(comm, "Metric SPD enforcement L2 error: %.4f%%\n", 100*errornorm);CHKERRQ(ierr);
      if (errornorm > tol) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric SPD enforcement test failed");
    }
    ierr = VecDestroy(&metric1);CHKERRQ(ierr);
    ierr = VecGetDM(determinant, &dmDet);CHKERRQ(ierr);
    ierr = VecDestroy(&determinant);CHKERRQ(ierr);
    ierr = DMDestroy(&dmDet);CHKERRQ(ierr);

    /* Test metric normalization */
    ierr = DMPlexMetricNormalize(dm, metric, PETSC_TRUE, PETSC_TRUE, &metric1);CHKERRQ(ierr);
    if (isotropic) {
      PetscReal target;

      ierr = DMPlexMetricGetTargetComplexity(dm, &target);CHKERRQ(ierr);
      scaling = PetscPowReal(target, 2.0/dim);
      ierr = DMPlexMetricCreateUniform(dm, 0, scaling, &metric2);CHKERRQ(ierr);
      ierr = VecAXPY(metric2, -1, metric1);CHKERRQ(ierr);
      ierr = VecNorm(metric2, NORM_2, &errornorm);CHKERRQ(ierr);
      errornorm /= norm;
      ierr = PetscPrintf(comm, "Metric normalization L2 error: %.4f%%\n", 100*errornorm);CHKERRQ(ierr);
      if (errornorm > tol) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric normalization test failed");
    }
    ierr = VecCopy(metric1, metric);CHKERRQ(ierr);
    ierr = VecDestroy(&metric2);CHKERRQ(ierr);
    ierr = VecDestroy(&metric1);CHKERRQ(ierr);
  }

  /* Adapt the mesh */
  ierr = DMAdaptMetric(dm, metric, bdLabel, rgLabel, &dmAdapt);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmAdapt, "DM_adapted");CHKERRQ(ierr);
  ierr = VecDestroy(&metric);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmAdapt, NULL, "-adapted_mesh_view");CHKERRQ(ierr);

  /* Test tag preservation */
  if (!noTagging) {
    PetscBool hasTag;
    PetscInt  size;

    ierr = DMGetLabel(dmAdapt, "Face Sets", &bdLabel);CHKERRQ(ierr);
    ierr = DMLabelHasStratum(bdLabel, 1, &hasTag);CHKERRQ(ierr);
    if (!hasTag) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have face tag 1");
    ierr = DMLabelHasStratum(bdLabel, 2, &hasTag);CHKERRQ(ierr);
    if (!hasTag) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have face tag 2");
    ierr = DMLabelGetNumValues(bdLabel, &size);CHKERRQ(ierr);
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh has the wrong number of face tags (got %d, expected 2)", size);

    ierr = DMGetLabel(dmAdapt, "Cell Sets", &rgLabel);CHKERRQ(ierr);
    ierr = DMLabelHasStratum(rgLabel, 3, &hasTag);CHKERRQ(ierr);
    if (!hasTag) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have cell tag 3");
    ierr = DMLabelHasStratum(rgLabel, 4, &hasTag);CHKERRQ(ierr);
    if (!hasTag) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh does not have cell tag 4");
    ierr = DMLabelGetNumValues(rgLabel, &size);CHKERRQ(ierr);
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Adapted mesh has the wrong number of cell tags (got %d, expected 2)", size);
  }

  /* Clean up */
  ierr = DMDestroy(&dmAdapt);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/*TEST

  testset:
    requires: pragmatic
    args: -dm_plex_metric_target_complexity 100 -dm_adaptor pragmatic -noTagging -dim 2

    test:
      suffix: uniform_2d_pragmatic
      args: -uniform -isotropic
    test:
      suffix: iso_2d_pragmatic
      args: -isotropic
    test:
      suffix: hessian_2d_pragmatic

  testset:
    requires: pragmatic tetgen
    args: -dm_plex_metric_target_complexity 100 -dm_adaptor pragmatic -noTagging -dim 3

    test:
      suffix: uniform_3d_pragmatic
      args: -uniform -isotropic -noTagging
    test:
      suffix: iso_3d_pragmatic
      args: -isotropic -noTagging
    test:
      suffix: hessian_3d_pragmatic

  testset:
    requires: mmg
    args: -dm_plex_metric_target_complexity 100 -dm_adaptor mmg -dim 2

    test:
      suffix: uniform_2d_mmg
      args: -uniform -isotropic
    test:
      suffix: iso_2d_mmg
      args: -isotropic
    test:
      suffix: hessian_2d_mmg

  testset:
    requires: mmg tetgen
    args: -dm_plex_metric_target_complexity 100 -dm_adaptor mmg -dim 3

    test:
      suffix: uniform_3d_mmg
      args: -uniform -isotropic
    test:
      suffix: iso_3d_mmg
      args: -isotropic
    test:
      suffix: hessian_3d_mmg

  testset:
    requires: parmmg tetgen
    nsize: 2
    args: -dm_plex_metric_target_complexity 100 -dm_adaptor parmmg -dim 3

    test:
      suffix: uniform_3d_parmmg
      args: -uniform -isotropic
    test:
      suffix: iso_3d_parmmg
      args: -isotropic
    test:
      suffix: hessian_3d_parmmg

TEST*/
