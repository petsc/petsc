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
  DMLabel         bdLabel = NULL;
  MPI_Comm        comm;
  PetscBool       uniform = PETSC_FALSE, isotropic = PETSC_FALSE;
  PetscErrorCode  ierr;
  PetscInt       *faces, dim = 3, numEdges = 4, d;
  PetscReal       scaling = 1.0;
  Vec             metric;

  /* Set up */
  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Mesh adaptation options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex60.c", dim, &dim, NULL, 2, 3);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_edges", "Number of edges on each boundary of the initial mesh", "ex60.c", numEdges, &numEdges, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-uniform", "Should the metric be assumed uniform?", "ex60.c", uniform, &uniform, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-isotropic", "Should the metric be assumed isotropic, or computed as a recovered Hessian?", "ex60.c", isotropic, &isotropic, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  /* Create box mesh */
  ierr = PetscMalloc1(dim, &faces);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) faces[d] = numEdges;
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
    PetscReal errornorm, norm, tol = 1.0e-10, weights[2] = {0.8, 0.2};
    Vec       metric1, metric2, metricComb;
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
    if (errornorm > tol) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric average test failed (L2 error %f)", errornorm);
    ierr = VecDestroy(&metricComb);CHKERRQ(ierr);

    /* Test metric intersection */
    if (isotropic) {
      ierr = DMPlexMetricIntersection(dm, 2, metrics, &metricComb);CHKERRQ(ierr);
      ierr = VecAXPY(metricComb, -1, metric1);CHKERRQ(ierr);
      ierr = VecNorm(metricComb, NORM_2, &errornorm);CHKERRQ(ierr);
      errornorm /= norm;
      if (errornorm > tol) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric intersection test failed (L2 error %f)", errornorm);
    }
    ierr = VecDestroy(&metric2);CHKERRQ(ierr);
    ierr = VecDestroy(&metricComb);CHKERRQ(ierr);
    ierr = VecCopy(metric, metric1);CHKERRQ(ierr);

    /* Test metric SPD enforcement */
    ierr = DMPlexMetricEnforceSPD(dm, PETSC_TRUE, PETSC_TRUE, metric);CHKERRQ(ierr);
    if (isotropic) {
      ierr = VecAXPY(metric1, -1, metric);CHKERRQ(ierr);
      ierr = VecNorm(metric1, NORM_2, &errornorm);CHKERRQ(ierr);
      errornorm /= norm;
      if (errornorm > tol) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric SPD enforcement test failed (L2 error %f)", errornorm);
    }
    ierr = VecDestroy(&metric1);CHKERRQ(ierr);

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
      if (errornorm > tol) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Metric normalization test failed (L2 error %f)", errornorm);
    }
    ierr = VecCopy(metric1, metric);CHKERRQ(ierr);
    ierr = VecDestroy(&metric2);CHKERRQ(ierr);
    ierr = VecDestroy(&metric1);CHKERRQ(ierr);
  }

  /* Adapt the mesh */
  ierr = DMAdaptMetric(dm, metric, bdLabel, &dmAdapt);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmAdapt, "DM_adapted");CHKERRQ(ierr);
  ierr = VecDestroy(&metric);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmAdapt, NULL, "-adapted_mesh_view");CHKERRQ(ierr);

  /* Compare DMs */
  ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMView(dmAdapt, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Clean up */
  ierr = DMDestroy(&dmAdapt);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/*TEST

  build:
    requires: pragmatic

  test:
    suffix: uniform_2d
    args: -dm_plex_metric_target_complexity 100 -dim 2 -uniform -isotropic
  test:
    suffix: uniform_3d
    args: -dm_plex_metric_target_complexity 100 -dim 3 -uniform -isotropic
  test:
    suffix: iso_2d
    args: -dm_plex_metric_target_complexity 100 -dim 2 -isotropic
  test:
    suffix: iso_3d
    args: -dm_plex_metric_target_complexity 100 -dim 3 -isotropic
  test:
    suffix: hessian_2d
    args: -dm_plex_metric_target_complexity 100 -dim 2
  test:
    suffix: hessian_3d
    args: -dm_plex_metric_target_complexity 100 -dim 3

TEST*/
