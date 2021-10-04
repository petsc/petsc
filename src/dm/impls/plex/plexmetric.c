#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscblaslapack.h>

PetscErrorCode DMPlexMetricSetFromOptions(DM dm)
{
  MPI_Comm       comm;
  PetscBool      isotropic = PETSC_FALSE, restrictAnisotropyFirst = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscReal      h_min = 1.0e-30, h_max = 1.0e+30, a_max = 1.0e+05, p = 1.0, target = 1000.0;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Riemannian metric options", "DMPlexMetric");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_metric_isotropic", "Is the metric isotropic?", "DMPlexMetricCreateIsotropic", isotropic, &isotropic, NULL);CHKERRQ(ierr);
  ierr = DMPlexMetricSetIsotropic(dm, isotropic);
  ierr = PetscOptionsBool("-dm_plex_metric_restrict_anisotropy_first", "Should anisotropy be restricted before normalization?", "DMPlexNormalize", restrictAnisotropyFirst, &restrictAnisotropyFirst, NULL);CHKERRQ(ierr);
  ierr = DMPlexMetricSetRestrictAnisotropyFirst(dm, restrictAnisotropyFirst);
  ierr = PetscOptionsReal("-dm_plex_metric_h_min", "Minimum tolerated metric magnitude", "DMPlexMetricEnforceSPD", h_min, &h_min, NULL);CHKERRQ(ierr);
  ierr = DMPlexMetricSetMinimumMagnitude(dm, h_min);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_plex_metric_h_max", "Maximum tolerated metric magnitude", "DMPlexMetricEnforceSPD", h_max, &h_max, NULL);CHKERRQ(ierr);
  ierr = DMPlexMetricSetMaximumMagnitude(dm, h_max);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_plex_metric_a_max", "Maximum tolerated anisotropy", "DMPlexMetricEnforceSPD", a_max, &a_max, NULL);CHKERRQ(ierr);
  ierr = DMPlexMetricSetMaximumAnisotropy(dm, a_max);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_plex_metric_p", "L-p normalization order", "DMPlexMetricNormalize", p, &p, NULL);CHKERRQ(ierr);
  ierr = DMPlexMetricSetNormalizationOrder(dm, p);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_plex_metric_target_complexity", "Target metric complexity", "DMPlexMetricNormalize", target, &target, NULL);CHKERRQ(ierr);
  ierr = DMPlexMetricSetTargetComplexity(dm, target);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricSetIsotropic - Record whether a metric is isotropic

  Input parameters:
+ dm        - The DM
- isotropic - Is the metric isotropic?

  Level: beginner

.seealso: DMPlexMetricIsIsotropic(), DMPlexMetricSetRestrictAnisotropyFirst()
*/
PetscErrorCode DMPlexMetricSetIsotropic(DM dm, PetscBool isotropic)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  plex->metricCtx->isotropic = isotropic;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricIsIsotropic - Is a metric is isotropic?

  Input parameters:
. dm        - The DM

  Output parameters:
. isotropic - Is the metric isotropic?

  Level: beginner

.seealso: DMPlexMetricSetIsotropic(), DMPlexMetricRestrictAnisotropyFirst()
*/
PetscErrorCode DMPlexMetricIsIsotropic(DM dm, PetscBool *isotropic)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  *isotropic = plex->metricCtx->isotropic;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricSetRestrictAnisotropyFirst - Record whether anisotropy should be restricted before normalization

  Input parameters:
+ dm                      - The DM
- restrictAnisotropyFirst - Should anisotropy be normalized first?

  Level: beginner

.seealso: DMPlexMetricSetIsotropic(), DMPlexMetricRestrictAnisotropyFirst()
*/
PetscErrorCode DMPlexMetricSetRestrictAnisotropyFirst(DM dm, PetscBool restrictAnisotropyFirst)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  plex->metricCtx->restrictAnisotropyFirst = restrictAnisotropyFirst;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricRestrictAnisotropyFirst - Is anisotropy restricted before normalization or after?

  Input parameters:
. dm                      - The DM

  Output parameters:
. restrictAnisotropyFirst - Is anisotropy be normalized first?

  Level: beginner

.seealso: DMPlexMetricIsIsotropic(), DMPlexMetricSetRestrictAnisotropyFirst()
*/
PetscErrorCode DMPlexMetricRestrictAnisotropyFirst(DM dm, PetscBool *restrictAnisotropyFirst)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  *restrictAnisotropyFirst = plex->metricCtx->restrictAnisotropyFirst;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricSetMinimumMagnitude - Set the minimum tolerated metric magnitude

  Input parameters:
+ dm    - The DM
- h_min - The minimum tolerated metric magnitude

  Level: beginner

.seealso: DMPlexMetricGetMinimumMagnitude(), DMPlexMetricSetMaximumMagnitude()
*/
PetscErrorCode DMPlexMetricSetMinimumMagnitude(DM dm, PetscReal h_min)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  if (h_min <= 0.0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Metric magnitudes must be positive, not %.4e", h_min);
  plex->metricCtx->h_min = h_min;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricGetMinimumMagnitude - Get the minimum tolerated metric magnitude

  Input parameters:
. dm    - The DM

  Input parameters:
. h_min - The minimum tolerated metric magnitude

  Level: beginner

.seealso: DMPlexMetricSetMinimumMagnitude(), DMPlexMetricGetMaximumMagnitude()
*/
PetscErrorCode DMPlexMetricGetMinimumMagnitude(DM dm, PetscReal *h_min)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  *h_min = plex->metricCtx->h_min;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricSetMaximumMagnitude - Set the maximum tolerated metric magnitude

  Input parameters:
+ dm    - The DM
- h_max - The maximum tolerated metric magnitude

  Level: beginner

.seealso: DMPlexMetricGetMaximumMagnitude(), DMPlexMetricSetMinimumMagnitude()
*/
PetscErrorCode DMPlexMetricSetMaximumMagnitude(DM dm, PetscReal h_max)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  if (h_max <= 0.0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Metric magnitudes must be positive, not %.4e", h_max);
  plex->metricCtx->h_max = h_max;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricGetMaximumMagnitude - Get the maximum tolerated metric magnitude

  Input parameters:
. dm    - The DM

  Input parameters:
. h_max - The maximum tolerated metric magnitude

  Level: beginner

.seealso: DMPlexMetricSetMaximumMagnitude(), DMPlexMetricGetMinimumMagnitude()
*/
PetscErrorCode DMPlexMetricGetMaximumMagnitude(DM dm, PetscReal *h_max)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  *h_max = plex->metricCtx->h_max;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricSetMaximumAnisotropy - Set the maximum tolerated metric anisotropy

  Input parameters:
+ dm    - The DM
- a_max - The maximum tolerated metric anisotropy

  Level: beginner

  Note: If the value zero is given then anisotropy will not be restricted. Otherwise, it should be at least one.

.seealso: DMPlexMetricGetMaximumAnisotropy(), DMPlexMetricSetMaximumMagnitude()
*/
PetscErrorCode DMPlexMetricSetMaximumAnisotropy(DM dm, PetscReal a_max)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  if (a_max < 1.0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Anisotropy must be at least one, not %.4e", a_max);
  plex->metricCtx->a_max = a_max;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricGetMaximumAnisotropy - Get the maximum tolerated metric anisotropy

  Input parameters:
. dm    - The DM

  Input parameters:
. a_max - The maximum tolerated metric anisotropy

  Level: beginner

.seealso: DMPlexMetricSetMaximumAnisotropy(), DMPlexMetricGetMaximumMagnitude()
*/
PetscErrorCode DMPlexMetricGetMaximumAnisotropy(DM dm, PetscReal *a_max)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  *a_max = plex->metricCtx->a_max;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricSetTargetComplexity - Set the target metric complexity

  Input parameters:
+ dm               - The DM
- targetComplexity - The target metric complexity

  Level: beginner

.seealso: DMPlexMetricGetTargetComplexity(), DMPlexMetricSetNormalizationOrder()
*/
PetscErrorCode DMPlexMetricSetTargetComplexity(DM dm, PetscReal targetComplexity)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  if (targetComplexity <= 0.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Metric complexity must be positive");
  plex->metricCtx->targetComplexity = targetComplexity;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricGetTargetComplexity - Get the target metric complexity

  Input parameters:
. dm               - The DM

  Input parameters:
. targetComplexity - The target metric complexity

  Level: beginner

.seealso: DMPlexMetricSetTargetComplexity(), DMPlexMetricGetNormalizationOrder()
*/
PetscErrorCode DMPlexMetricGetTargetComplexity(DM dm, PetscReal *targetComplexity)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  *targetComplexity = plex->metricCtx->targetComplexity;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricSetNormalizationOrder - Set the order p for L-p normalization

  Input parameters:
+ dm - The DM
- p  - The normalization order

  Level: beginner

.seealso: DMPlexMetricGetNormalizationOrder(), DMPlexMetricSetTargetComplexity()
*/
PetscErrorCode DMPlexMetricSetNormalizationOrder(DM dm, PetscReal p)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  if (p < 1.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Normalization order must be one or greater");
  plex->metricCtx->p = p;
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricGetNormalizationOrder - Get the order p for L-p normalization

  Input parameters:
. dm - The DM

  Input parameters:
. p - The normalization order

  Level: beginner

.seealso: DMPlexMetricSetNormalizationOrder(), DMPlexMetricGetTargetComplexity()
*/
PetscErrorCode DMPlexMetricGetNormalizationOrder(DM dm, PetscReal *p)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  *p = plex->metricCtx->p;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexP1FieldCreate_Private(DM dm, PetscInt f, PetscInt size, Vec *metric)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFE        fe;
  PetscInt       dim;

  PetscFunctionBegin;

  /* Extract metadata from dm */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  /* Create a P1 field of the requested size */
  ierr = PetscFECreateLagrange(comm, dim, size, PETSC_TRUE, 1, PETSC_DETERMINE, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dm, f, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, metric);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  DMPlexMetricCreate - Create a Riemannian metric field

  Input parameters:
+ dm     - The DM
- f      - The field number to use

  Output parameter:
. metric - The metric

  Level: beginner

  Notes:

  It is assumed that the DM is comprised of simplices.

  Command line options for Riemannian metrics:

  -dm_plex_metric_isotropic                 - Is the metric isotropic?
  -dm_plex_metric_restrict_anisotropy_first - Should anisotropy be restricted before normalization?
  -dm_plex_metric_h_min                     - Minimum tolerated metric magnitude
  -dm_plex_metric_h_max                     - Maximum tolerated metric magnitude
  -dm_plex_metric_a_max                     - Maximum tolerated anisotropy
  -dm_plex_metric_p                         - L-p normalization order
  -dm_plex_metric_target_complexity         - Target metric complexity

.seealso: DMPlexMetricCreateUniform(), DMPlexMetricCreateIsotropic()
*/
PetscErrorCode DMPlexMetricCreate(DM dm, PetscInt f, Vec *metric)
{
  DM_Plex       *plex = (DM_Plex *) dm->data;
  PetscErrorCode ierr;
  PetscInt       coordDim, Nd;

  PetscFunctionBegin;
  if (!plex->metricCtx) {
    ierr = PetscNew(&plex->metricCtx);CHKERRQ(ierr);
    ierr = DMPlexMetricSetFromOptions(dm);CHKERRQ(ierr);
  }
  ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
  Nd = coordDim*coordDim;
  ierr = DMPlexP1FieldCreate_Private(dm, f, Nd, metric);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal scaling;  /* Scaling for uniform metric diagonal */
} DMPlexMetricUniformCtx;

static PetscErrorCode diagonal(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  DMPlexMetricUniformCtx *user = (DMPlexMetricUniformCtx*)ctx;
  PetscInt                i, j;

  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      if (i == j) u[i+dim*j] = user->scaling;
      else u[i+dim*j] = 0.0;
    }
  }
  return 0;
}

/*
  DMPlexMetricCreateUniform - Construct a uniform isotropic metric

  Input parameters:
+ dm     - The DM
. f      - The field number to use
- alpha  - Scaling parameter for the diagonal

  Output parameter:
. metric - The uniform metric

  Level: beginner

  Note: It is assumed that the DM is comprised of simplices.

.seealso: DMPlexMetricCreate(), DMPlexMetricCreateIsotropic()
*/
PetscErrorCode DMPlexMetricCreateUniform(DM dm, PetscInt f, PetscReal alpha, Vec *metric)
{
  DMPlexMetricUniformCtx user;
  PetscErrorCode       (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  PetscErrorCode         ierr;
  void                  *ctxs[1];

  PetscFunctionBegin;
  ierr = DMPlexMetricCreate(dm, f, metric);CHKERRQ(ierr);
  if (!alpha) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Uniform metric scaling is undefined");
  if (alpha < 1.0e-30) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Uniform metric scaling %e should be positive", alpha);
  else user.scaling = alpha;
  funcs[0] = diagonal;
  ctxs[0] = &user;
  ierr = DMProjectFunctionLocal(dm, 0.0, funcs, ctxs, INSERT_ALL_VALUES, *metric);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricCreateIsotropic - Construct an isotropic metric from an error indicator

  Input parameters:
+ dm        - The DM
. f         - The field number to use
- indicator - The error indicator

  Output parameter:
. metric    - The isotropic metric

  Level: beginner

  Notes:

  It is assumed that the DM is comprised of simplices.

  The indicator needs to be a scalar field defined at *vertices*.

.seealso: DMPlexMetricCreate(), DMPlexMetricCreateUniform()
*/
PetscErrorCode DMPlexMetricCreateIsotropic(DM dm, PetscInt f, Vec indicator, Vec *metric)
{
  DM                 dmIndi;
  PetscErrorCode     ierr;
  PetscInt           dim, d, vStart, vEnd, v;
  const PetscScalar *indi;
  PetscScalar       *met;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexMetricCreate(dm, f, metric);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(indicator, &indi);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(*metric, &met);CHKERRQ(ierr);
  ierr = VecGetDM(indicator, &dmIndi);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *vindi, *vmet;
    ierr = DMPlexPointLocalRead(dmIndi, v, indi, &vindi);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRef(dm, v, met, &vmet);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) vmet[d*(dim+1)] = vindi[0];
  }
  ierr = VecRestoreArrayWrite(*metric, &met);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(indicator, &indi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMetricModify_Private(PetscInt dim, PetscReal h_min, PetscReal h_max, PetscReal a_max, PetscScalar Mp[])
{
  PetscErrorCode ierr;
  PetscInt       i, j, k;
  PetscReal     *eigs, max_eig, l_min = 1.0/(h_max*h_max), l_max = 1.0/(h_min*h_min), la_min = 1.0/(a_max*a_max);
  PetscScalar   *Mpos;

  PetscFunctionBegin;
  ierr = PetscMalloc2(dim*dim, &Mpos, dim, &eigs);CHKERRQ(ierr);

  /* Symmetrize */
  for (i = 0; i < dim; ++i) {
    Mpos[i*dim+i] = Mp[i*dim+i];
    for (j = i+1; j < dim; ++j) {
      Mpos[i*dim+j] = 0.5*(Mp[i*dim+j] + Mp[j*dim+i]);
      Mpos[j*dim+i] = Mpos[i*dim+j];
    }
  }

  /* Compute eigendecomposition */
  {
    PetscScalar  *work;
    PetscBLASInt lwork;

    lwork = 5*dim;
    ierr = PetscMalloc1(5*dim, &work);CHKERRQ(ierr);
    {
      PetscBLASInt lierr;
      PetscBLASInt nb;

      ierr = PetscBLASIntCast(dim, &nb);CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      {
        PetscReal *rwork;
        ierr = PetscMalloc1(3*dim, &rwork);CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&nb,Mpos,&nb,eigs,work,&lwork,rwork,&lierr));
        ierr = PetscFree(rwork);CHKERRQ(ierr);
      }
#else
      PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&nb,Mpos,&nb,eigs,work,&lwork,&lierr));
#endif
      if (lierr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine %d", (int) lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
    }
    ierr = PetscFree(work);CHKERRQ(ierr);
  }

  /* Reflect to positive orthant and enforce maximum and minimum size */
  max_eig = 0.0;
  for (i = 0; i < dim; ++i) {
    eigs[i] = PetscMin(l_max, PetscMax(l_min, PetscAbsReal(eigs[i])));
    max_eig = PetscMax(eigs[i], max_eig);
  }

  /* Enforce maximum anisotropy */
  for (i = 0; i < dim; ++i) {
    if (a_max > 1.0) eigs[i] = PetscMax(eigs[i], max_eig*la_min);
  }

  /* Reconstruct Hessian */
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      Mp[i*dim+j] = 0.0;
      for (k = 0; k < dim; ++k) {
        Mp[i*dim+j] += Mpos[k*dim+i] * eigs[k] * Mpos[k*dim+j];
      }
    }
  }
  ierr = PetscFree2(Mpos, eigs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  DMPlexMetricEnforceSPD - Enforce symmetric positive-definiteness of a metric

  Input parameters:
+ dm                 - The DM
. restrictSizes      - Should maximum/minimum metric magnitudes be enforced?
. restrictAnisotropy - Should maximum anisotropy be enforced?
- metric             - The metric

  Output parameter:
. metric             - The metric

  Level: beginner

.seealso: DMPlexMetricNormalize(), DMPlexMetricIntersection()
*/
PetscErrorCode DMPlexMetricEnforceSPD(DM dm, PetscBool restrictSizes, PetscBool restrictAnisotropy, Vec metric)
{
  PetscErrorCode ierr;
  PetscInt       dim, vStart, vEnd, v;
  PetscScalar   *met;
  PetscReal      h_min = 1.0e-30, h_max = 1.0e+30, a_max = 0.0;

  PetscFunctionBegin;

  /* Extract metadata from dm */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (restrictSizes) {
    ierr = DMPlexMetricGetMinimumMagnitude(dm, &h_min);CHKERRQ(ierr);
    ierr = DMPlexMetricGetMaximumMagnitude(dm, &h_max);CHKERRQ(ierr);
    h_min = PetscMax(h_min, 1.0e-30);
    h_max = PetscMin(h_max, 1.0e+30);
    if (h_min >= h_max) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Incompatible min/max metric magnitudes (%.4e not smaller than %.4e)", h_min, h_max);
  }
  if (restrictAnisotropy) {
    ierr = DMPlexMetricGetMaximumAnisotropy(dm, &a_max);CHKERRQ(ierr);
    a_max = PetscMin(a_max, 1.0e+30);
  }

  /* Enforce SPD */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = VecGetArray(metric, &met);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *vmet;
    ierr = DMPlexPointLocalRef(dm, v, met, &vmet);CHKERRQ(ierr);
    ierr = DMPlexMetricModify_Private(dim, h_min, h_max, a_max, vmet);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(metric, &met);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static void detMFunc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar p = constants[0];
  PetscReal         detH = 0.0;

  if      (dim == 2) DMPlex_Det2D_Scalar_Internal(&detH, u);
  else if (dim == 3) DMPlex_Det3D_Scalar_Internal(&detH, u);
  f0[0] = PetscPowReal(detH, p/(2.0*p + dim));
}

/*
  DMPlexMetricNormalize - Apply L-p normalization to a metric

  Input parameters:
+ dm                 - The DM
. metricIn           - The unnormalized metric
. restrictSizes      - Should maximum/minimum metric magnitudes be enforced?
- restrictAnisotropy - Should maximum metric anisotropy be enforced?

  Output parameter:
. metricOut          - The normalized metric

  Level: beginner

.seealso: DMPlexMetricEnforceSPD(), DMPlexMetricIntersection()
*/
PetscErrorCode DMPlexMetricNormalize(DM dm, Vec metricIn, PetscBool restrictSizes, PetscBool restrictAnisotropy, Vec *metricOut)
{
  MPI_Comm         comm;
  PetscBool        restrictAnisotropyFirst;
  PetscDS          ds;
  PetscErrorCode   ierr;
  PetscInt         dim, Nd, vStart, vEnd, v, i;
  PetscScalar     *met, integral, constants[1];
  PetscReal        p, h_min = 1.0e-30, h_max = 1.0e+30, a_max = 0.0, factGlob, target;

  PetscFunctionBegin;

  /* Extract metadata from dm */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  Nd = dim*dim;

  /* Set up metric and ensure it is SPD */
  ierr = DMPlexMetricCreate(dm, 0, metricOut);CHKERRQ(ierr);
  ierr = VecCopy(metricIn, *metricOut);CHKERRQ(ierr);
  ierr = DMPlexMetricRestrictAnisotropyFirst(dm, &restrictAnisotropyFirst);CHKERRQ(ierr);
  ierr = DMPlexMetricEnforceSPD(dm, PETSC_FALSE, (PetscBool)(restrictAnisotropy && restrictAnisotropyFirst), *metricOut);CHKERRQ(ierr);

  /* Compute global normalization factor */
  ierr = DMPlexMetricGetTargetComplexity(dm, &target);CHKERRQ(ierr);
  ierr = DMPlexMetricGetNormalizationOrder(dm, &p);CHKERRQ(ierr);
  constants[0] = p;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetConstants(ds, 1, constants);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(ds, 0, detMFunc);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, *metricOut, &integral, NULL);CHKERRQ(ierr);
  factGlob = PetscPowReal(target/PetscRealPart(integral), 2.0/dim);

  /* Apply local scaling */
  if (restrictSizes) {
    ierr = DMPlexMetricGetMinimumMagnitude(dm, &h_min);CHKERRQ(ierr);
    ierr = DMPlexMetricGetMaximumMagnitude(dm, &h_max);CHKERRQ(ierr);
    h_min = PetscMax(h_min, 1.0e-30);
    h_max = PetscMin(h_max, 1.0e+30);
    if (h_min >= h_max) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Incompatible min/max metric magnitudes (%.4e not smaller than %.4e)", h_min, h_max);
  }
  if (restrictAnisotropy && !restrictAnisotropyFirst) {
    ierr = DMPlexMetricGetMaximumAnisotropy(dm, &a_max);CHKERRQ(ierr);
    a_max = PetscMin(a_max, 1.0e+30);
  }
  ierr = VecGetArray(*metricOut, &met);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscScalar *Mp;
    PetscReal    detM, fact;

    ierr = DMPlexPointLocalRef(dm, v, met, &Mp);CHKERRQ(ierr);
    if      (dim == 2) DMPlex_Det2D_Scalar_Internal(&detM, Mp);
    else if (dim == 3) DMPlex_Det3D_Scalar_Internal(&detM, Mp);
    else SETERRQ1(comm, PETSC_ERR_SUP, "Dimension %d not supported", dim);
    fact = factGlob * PetscPowReal(PetscAbsReal(detM), -1.0/(2*p+dim));
    for (i = 0; i < Nd; ++i) Mp[i] *= fact;
    if (restrictSizes) { ierr = DMPlexMetricModify_Private(dim, h_min, h_max, a_max, Mp);CHKERRQ(ierr); }
  }
  ierr = VecRestoreArray(*metricOut, &met);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  DMPlexMetricAverage - Compute the average of a list of metrics

  Input Parameter:
+ dm         - The DM
. numMetrics - The number of metrics to be averaged
. weights    - Weights for the average
- metrics    - The metrics to be averaged

  Output Parameter:
. metricAvg  - The averaged metric

  Level: beginner

  Notes:
  The weights should sum to unity.

  If weights are not provided then an unweighted average is used.

.seealso: DMPlexMetricAverage2(), DMPlexMetricAverage3(), DMPlexMetricIntersection()
*/
PetscErrorCode DMPlexMetricAverage(DM dm, PetscInt numMetrics, PetscReal weights[], Vec metrics[], Vec *metricAvg)
{
  PetscBool      haveWeights = PETSC_TRUE;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      sum = 0.0, tol = 1.0e-10;

  PetscFunctionBegin;
  if (numMetrics < 1) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot average %d < 1 metrics", numMetrics); }
  ierr = DMPlexMetricCreate(dm, 0, metricAvg);CHKERRQ(ierr);
  ierr = VecSet(*metricAvg, 0.0);CHKERRQ(ierr);

  /* Default to the unweighted case */
  if (!weights) {
    ierr = PetscMalloc1(numMetrics, &weights);CHKERRQ(ierr);
    haveWeights = PETSC_FALSE;
    for (i = 0; i < numMetrics; ++i) {weights[i] = 1.0/numMetrics; }
  }

  /* Check weights sum to unity */
  for (i = 0; i < numMetrics; ++i) { sum += weights[i]; }
  if (PetscAbsReal(sum - 1) > tol) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Weights do not sum to unity"); }

  /* Compute metric average */
  for (i = 0; i < numMetrics; ++i) { ierr = VecAXPY(*metricAvg, weights[i], metrics[i]);CHKERRQ(ierr); }
  if (!haveWeights) {ierr = PetscFree(weights); }
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricAverage2 - Compute the unweighted average of two metrics

  Input Parameter:
+ dm         - The DM
. metric1    - The first metric to be averaged
- metric2    - The second metric to be averaged

  Output Parameter:
. metricAvg  - The averaged metric

  Level: beginner

.seealso: DMPlexMetricAverage(), DMPlexMetricAverage3()
*/
PetscErrorCode DMPlexMetricAverage2(DM dm, Vec metric1, Vec metric2, Vec *metricAvg)
{
  PetscErrorCode ierr;
  PetscReal      weights[2] = {0.5, 0.5};
  Vec            metrics[2] = {metric1, metric2};

  PetscFunctionBegin;
  ierr = DMPlexMetricAverage(dm, 2, weights, metrics, metricAvg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricAverage3 - Compute the unweighted average of three metrics

  Input Parameter:
+ dm         - The DM
. metric1    - The first metric to be averaged
. metric2    - The second metric to be averaged
- metric3    - The third metric to be averaged

  Output Parameter:
. metricAvg  - The averaged metric

  Level: beginner

.seealso: DMPlexMetricAverage(), DMPlexMetricAverage2()
*/
PetscErrorCode DMPlexMetricAverage3(DM dm, Vec metric1, Vec metric2, Vec metric3, Vec *metricAvg)
{
  PetscErrorCode ierr;
  PetscReal      weights[3] = {1.0/3.0, 1.0/3.0, 1.0/3.0};
  Vec            metrics[3] = {metric1, metric2, metric3};

  PetscFunctionBegin;
  ierr = DMPlexMetricAverage(dm, 3, weights, metrics, metricAvg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexMetricIntersection_Private(PetscInt dim, PetscScalar M1[], PetscScalar M2[])
{
  PetscErrorCode ierr;
  PetscInt       i, j, k, l, m;
  PetscReal     *evals, *evals1;
  PetscScalar   *evecs, *sqrtM1, *isqrtM1;

  PetscFunctionBegin;
  ierr = PetscMalloc5(dim*dim, &evecs, dim*dim, &sqrtM1, dim*dim, &isqrtM1, dim, &evals, dim, &evals1);CHKERRQ(ierr);
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      evecs[i*dim+j] = M1[i*dim+j];
    }
  }
  {
    PetscScalar *work;
    PetscBLASInt lwork;

    lwork = 5*dim;
    ierr = PetscMalloc1(5*dim, &work);CHKERRQ(ierr);
    {
      PetscBLASInt lierr, nb;
      PetscReal    sqrtk;

      /* Compute eigendecomposition of M1 */
      ierr = PetscBLASIntCast(dim, &nb);CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      {
        PetscReal *rwork;
        ierr = PetscMalloc1(3*dim, &rwork);CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &nb, evecs, &nb, evals1, work, &lwork, rwork, &lierr));
        ierr = PetscFree(rwork);CHKERRQ(ierr);
      }
#else
      PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &nb, evecs, &nb, evals1, work, &lwork, &lierr));
#endif
      if (lierr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine %d", (int) lierr);
      ierr = PetscFPTrapPop();

      /* Compute square root and reciprocal */
      for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
          sqrtM1[i*dim+j] = 0.0;
          isqrtM1[i*dim+j] = 0.0;
          for (k = 0; k < dim; ++k) {
            sqrtk = PetscSqrtReal(evals1[k]);
            sqrtM1[i*dim+j] += evecs[k*dim+i] * sqrtk * evecs[k*dim+j];
            isqrtM1[i*dim+j] += evecs[k*dim+i] * (1.0/sqrtk) * evecs[k*dim+j];
          }
        }
      }

      /* Map into the space spanned by the eigenvectors of M1 */
      for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
          evecs[i*dim+j] = 0.0;
          for (k = 0; k < dim; ++k) {
            for (l = 0; l < dim; ++l) {
              evecs[i*dim+j] += isqrtM1[i*dim+k] * M2[l*dim+k] * isqrtM1[j*dim+l];
            }
          }
        }
      }

      /* Compute eigendecomposition */
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      {
        PetscReal *rwork;
        ierr = PetscMalloc1(3*dim, &rwork);CHKERRQ(ierr);
        PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &nb, evecs, &nb, evals, work, &lwork, rwork, &lierr));
        ierr = PetscFree(rwork);CHKERRQ(ierr);
      }
#else
      PetscStackCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &nb, evecs, &nb, evals, work, &lwork, &lierr));
#endif
      if (lierr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine %d", (int) lierr);
      ierr = PetscFPTrapPop();

      /* Modify eigenvalues */
      for (i = 0; i < dim; ++i) evals[i] = PetscMin(evals[i], evals1[i]);

      /* Map back to get the intersection */
      for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
          M2[i*dim+j] = 0.0;
          for (k = 0; k < dim; ++k) {
            for (l = 0; l < dim; ++l) {
              for (m = 0; m < dim; ++m) {
                M2[i*dim+j] += sqrtM1[i*dim+k] * evecs[l*dim+k] * evals[l] * evecs[l*dim+m] * sqrtM1[j*dim+m];
              }
            }
          }
        }
      }
    }
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  ierr = PetscFree5(evecs, sqrtM1, isqrtM1, evals, evals1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricIntersection - Compute the intersection of a list of metrics

  Input Parameter:
+ dm         - The DM
. numMetrics - The number of metrics to be intersected
- metrics    - The metrics to be intersected

  Output Parameter:
. metricInt  - The intersected metric

  Level: beginner

  Notes:

  The intersection of a list of metrics has the maximal ellipsoid which fits within the ellipsoids of the component metrics.

  The implementation used here is only consistent with the maximal ellipsoid definition in the case numMetrics = 2.

.seealso: DMPlexMetricIntersection2(), DMPlexMetricIntersection3(), DMPlexMetricAverage()
*/
PetscErrorCode DMPlexMetricIntersection(DM dm, PetscInt numMetrics, Vec metrics[], Vec *metricInt)
{
  PetscErrorCode ierr;
  PetscInt       dim, vStart, vEnd, v, i;
  PetscScalar   *met, *meti, *M, *Mi;

  PetscFunctionBegin;
  if (numMetrics < 1) { SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot intersect %d < 1 metrics", numMetrics); }

  /* Extract metadata from dm */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexMetricCreate(dm, 0, metricInt);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);

  /* Copy over the first metric */
  ierr = VecCopy(metrics[0], *metricInt);CHKERRQ(ierr);

  /* Intersect subsequent metrics in turn */
  if (numMetrics > 1) {
    ierr = VecGetArray(*metricInt, &met);CHKERRQ(ierr);
    for (i = 1; i < numMetrics; ++i) {
      ierr = VecGetArray(metrics[i], &meti);CHKERRQ(ierr);
      for (v = vStart; v < vEnd; ++v) {
        ierr = DMPlexPointLocalRef(dm, v, met, &M);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRef(dm, v, meti, &Mi);CHKERRQ(ierr);
        ierr = DMPlexMetricIntersection_Private(dim, Mi, M);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(metrics[i], &meti);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(*metricInt, &met);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*
  DMPlexMetricIntersection2 - Compute the intersection of two metrics

  Input Parameter:
+ dm        - The DM
. metric1   - The first metric to be intersected
- metric2   - The second metric to be intersected

  Output Parameter:
. metricInt - The intersected metric

  Level: beginner

.seealso: DMPlexMetricIntersection(), DMPlexMetricIntersection3()
*/
PetscErrorCode DMPlexMetricIntersection2(DM dm, Vec metric1, Vec metric2, Vec *metricInt)
{
  PetscErrorCode ierr;
  Vec            metrics[2] = {metric1, metric2};

  PetscFunctionBegin;
  ierr = DMPlexMetricIntersection(dm, 2, metrics, metricInt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexMetricIntersection3 - Compute the intersection of three metrics

  Input Parameter:
+ dm        - The DM
. metric1   - The first metric to be intersected
. metric2   - The second metric to be intersected
- metric3   - The third metric to be intersected

  Output Parameter:
. metricInt - The intersected metric

  Level: beginner

.seealso: DMPlexMetricIntersection(), DMPlexMetricIntersection2()
*/
PetscErrorCode DMPlexMetricIntersection3(DM dm, Vec metric1, Vec metric2, Vec metric3, Vec *metricInt)
{
  PetscErrorCode ierr;
  Vec            metrics[3] = {metric1, metric2, metric3};

  PetscFunctionBegin;
  ierr = DMPlexMetricIntersection(dm, 3, metrics, metricInt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
