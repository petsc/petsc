/*
  Code for timestepping with BDF methods
*/
#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/
#include <petscdm.h>

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@book{Brenan1995,\n"
                               "  title     = {Numerical Solution of Initial-Value Problems in Differential-Algebraic Equations},\n"
                               "  author    = {Brenan, K. and Campbell, S. and Petzold, L.},\n"
                               "  publisher = {Society for Industrial and Applied Mathematics},\n"
                               "  year      = {1995},\n"
                               "  doi       = {10.1137/1.9781611971224},\n}\n";

typedef struct {
  PetscInt  k, n;
  PetscReal time[6 + 2];
  Vec       work[6 + 2];
  Vec       tvwork[6 + 2];
  PetscReal shift;
  Vec       vec_dot; /* Xdot when !transientvar, else Cdot where C(X) is the transient variable. */
  Vec       vec_wrk;
  Vec       vec_lte;

  PetscBool    transientvar;
  PetscInt     order;
  TSStepStatus status;
} TS_BDF;

/* Compute Lagrange polynomials on T[:n] evaluated at t.
 * If one has data (T[i], Y[i]), then the interpolation/extrapolation is f(t) = \sum_i L[i]*Y[i].
 */
static inline void LagrangeBasisVals(PetscInt n, PetscReal t, const PetscReal T[], PetscScalar L[])
{
  PetscInt k, j;
  for (k = 0; k < n; k++)
    for (L[k] = 1, j = 0; j < n; j++)
      if (j != k) L[k] *= (t - T[j]) / (T[k] - T[j]);
}

static inline void LagrangeBasisDers(PetscInt n, PetscReal t, const PetscReal T[], PetscScalar dL[])
{
  PetscInt k, j, i;
  for (k = 0; k < n; k++)
    for (dL[k] = 0, j = 0; j < n; j++)
      if (j != k) {
        PetscReal L = 1 / (T[k] - T[j]);
        for (i = 0; i < n; i++)
          if (i != j && i != k) L *= (t - T[i]) / (T[k] - T[i]);
        dL[k] += L;
      }
}

static PetscErrorCode TSBDF_GetVecs(TS ts, DM dm, Vec *Xdot, Vec *Ydot)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;

  PetscFunctionBegin;
  if (dm && dm != ts->dm) {
    PetscCall(DMGetNamedGlobalVector(dm, "TSBDF_Vec_Xdot", Xdot));
    PetscCall(DMGetNamedGlobalVector(dm, "TSBDF_Vec_Ydot", Ydot));
  } else {
    *Xdot = bdf->vec_dot;
    *Ydot = bdf->vec_wrk;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDF_RestoreVecs(TS ts, DM dm, Vec *Xdot, Vec *Ydot)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;

  PetscFunctionBegin;
  if (dm && dm != ts->dm) {
    PetscCall(DMRestoreNamedGlobalVector(dm, "TSBDF_Vec_Xdot", Xdot));
    PetscCall(DMRestoreNamedGlobalVector(dm, "TSBDF_Vec_Ydot", Ydot));
  } else {
    PetscCheck(*Xdot == bdf->vec_dot, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "Vec does not match the cache");
    PetscCheck(*Ydot == bdf->vec_wrk, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "Vec does not match the cache");
    *Xdot = NULL;
    *Ydot = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCoarsenHook_TSBDF(DM fine, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRestrictHook_TSBDF(DM fine, Mat restrct, Vec rscale, Mat inject, DM coarse, void *ctx)
{
  TS  ts = (TS)ctx;
  Vec Ydot, Ydot_c;
  Vec Xdot, Xdot_c;

  PetscFunctionBegin;
  PetscCall(TSBDF_GetVecs(ts, fine, &Xdot, &Ydot));
  PetscCall(TSBDF_GetVecs(ts, coarse, &Xdot_c, &Ydot_c));

  PetscCall(MatRestrict(restrct, Ydot, Ydot_c));
  PetscCall(VecPointwiseMult(Ydot_c, rscale, Ydot_c));

  PetscCall(TSBDF_RestoreVecs(ts, fine, &Xdot, &Ydot));
  PetscCall(TSBDF_RestoreVecs(ts, coarse, &Xdot_c, &Ydot_c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDF_Advance(TS ts, PetscReal t, Vec X)
{
  TS_BDF  *bdf = (TS_BDF *)ts->data;
  PetscInt i, n = (PetscInt)(sizeof(bdf->work) / sizeof(Vec));
  Vec      tail = bdf->work[n - 1], tvtail = bdf->tvwork[n - 1];

  PetscFunctionBegin;
  for (i = n - 1; i >= 2; i--) {
    bdf->time[i]   = bdf->time[i - 1];
    bdf->work[i]   = bdf->work[i - 1];
    bdf->tvwork[i] = bdf->tvwork[i - 1];
  }
  bdf->n         = PetscMin(bdf->n + 1, n - 1);
  bdf->time[1]   = t;
  bdf->work[1]   = tail;
  bdf->tvwork[1] = tvtail;
  PetscCall(VecCopy(X, tail));
  PetscCall(TSComputeTransientVariable(ts, tail, tvtail));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDF_VecLTE(TS ts, PetscInt order, Vec lte)
{
  TS_BDF     *bdf = (TS_BDF *)ts->data;
  PetscInt    i, n = order + 1;
  PetscReal  *time = bdf->time;
  Vec        *vecs = bdf->work;
  PetscScalar a[8], b[8], alpha[8];

  PetscFunctionBegin;
  LagrangeBasisDers(n + 0, time[0], time, a);
  a[n] = 0;
  LagrangeBasisDers(n + 1, time[0], time, b);
  for (i = 0; i < n + 1; i++) alpha[i] = (a[i] - b[i]) / a[0];
  PetscCall(VecZeroEntries(lte));
  PetscCall(VecMAXPY(lte, n + 1, alpha, vecs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDF_Extrapolate(TS ts, PetscInt order, PetscReal t, Vec X)
{
  TS_BDF     *bdf  = (TS_BDF *)ts->data;
  PetscInt    n    = order + 1;
  PetscReal  *time = bdf->time + 1;
  Vec        *vecs = bdf->work + 1;
  PetscScalar alpha[7];

  PetscFunctionBegin;
  n = PetscMin(n, bdf->n);
  LagrangeBasisVals(n, t, time, alpha);
  PetscCall(VecZeroEntries(X));
  PetscCall(VecMAXPY(X, n, alpha, vecs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDF_Interpolate(TS ts, PetscInt order, PetscReal t, Vec X)
{
  TS_BDF     *bdf  = (TS_BDF *)ts->data;
  PetscInt    n    = order + 1;
  PetscReal  *time = bdf->time;
  Vec        *vecs = bdf->work;
  PetscScalar alpha[7];

  PetscFunctionBegin;
  LagrangeBasisVals(n, t, time, alpha);
  PetscCall(VecZeroEntries(X));
  PetscCall(VecMAXPY(X, n, alpha, vecs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute the affine term V0 such that Xdot = shift*X + V0.
 *
 * When using transient variables, we're computing Cdot = shift*C(X) + V0, and thus choose a linear combination of tvwork.
 */
static PetscErrorCode TSBDF_PreSolve(TS ts)
{
  TS_BDF     *bdf = (TS_BDF *)ts->data;
  PetscInt    i, n = PetscMax(bdf->k, 1) + 1;
  Vec         V, V0;
  Vec         vecs[7];
  PetscScalar alpha[7];

  PetscFunctionBegin;
  PetscCall(TSBDF_GetVecs(ts, NULL, &V, &V0));
  LagrangeBasisDers(n, bdf->time[0], bdf->time, alpha);
  for (i = 1; i < n; i++) vecs[i] = bdf->transientvar ? bdf->tvwork[i] : bdf->work[i];
  PetscCall(VecZeroEntries(V0));
  PetscCall(VecMAXPY(V0, n - 1, alpha + 1, vecs + 1));
  bdf->shift = PetscRealPart(alpha[0]);
  PetscCall(TSBDF_RestoreVecs(ts, NULL, &V, &V0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDF_SNESSolve(TS ts, Vec b, Vec x)
{
  PetscInt nits, lits;

  PetscFunctionBegin;
  PetscCall(TSBDF_PreSolve(ts));
  PetscCall(SNESSolve(ts->snes, b, x));
  PetscCall(SNESGetIterationNumber(ts->snes, &nits));
  PetscCall(SNESGetLinearSolveIterations(ts->snes, &lits));
  ts->snes_its += nits;
  ts->ksp_its += lits;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDF_Restart(TS ts, PetscBool *accept)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;

  PetscFunctionBegin;
  bdf->k = 1;
  bdf->n = 0;
  PetscCall(TSBDF_Advance(ts, ts->ptime, ts->vec_sol));

  bdf->time[0] = ts->ptime + ts->time_step / 2;
  PetscCall(VecCopy(bdf->work[1], bdf->work[0]));
  PetscCall(TSPreStage(ts, bdf->time[0]));
  PetscCall(TSBDF_SNESSolve(ts, NULL, bdf->work[0]));
  PetscCall(TSPostStage(ts, bdf->time[0], 0, &bdf->work[0]));
  PetscCall(TSAdaptCheckStage(ts->adapt, ts, bdf->time[0], bdf->work[0], accept));
  if (!*accept) PetscFunctionReturn(PETSC_SUCCESS);

  bdf->k = PetscMin(2, bdf->order);
  bdf->n++;
  PetscCall(VecCopy(bdf->work[0], bdf->work[2]));
  bdf->time[2] = bdf->time[0];
  PetscCall(TSComputeTransientVariable(ts, bdf->work[2], bdf->tvwork[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *const BDF_SchemeName[] = {"", "1", "2", "3", "4", "5", "6"};

static PetscErrorCode TSStep_BDF(TS ts)
{
  TS_BDF   *bdf        = (TS_BDF *)ts->data;
  PetscInt  rejections = 0;
  PetscBool stageok, accept = PETSC_TRUE;
  PetscReal next_time_step = ts->time_step;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation, &cited));

  if (!ts->steprollback && !ts->steprestart) {
    bdf->k = PetscMin(bdf->k + 1, bdf->order);
    PetscCall(TSBDF_Advance(ts, ts->ptime, ts->vec_sol));
  }

  bdf->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && bdf->status != TS_STEP_COMPLETE) {
    if (ts->steprestart) {
      PetscCall(TSBDF_Restart(ts, &stageok));
      if (!stageok) goto reject_step;
    }

    bdf->time[0] = ts->ptime + ts->time_step;
    PetscCall(TSBDF_Extrapolate(ts, bdf->k - (accept ? 0 : 1), bdf->time[0], bdf->work[0]));
    PetscCall(TSPreStage(ts, bdf->time[0]));
    PetscCall(TSBDF_SNESSolve(ts, NULL, bdf->work[0]));
    PetscCall(TSPostStage(ts, bdf->time[0], 0, &bdf->work[0]));
    PetscCall(TSAdaptCheckStage(ts->adapt, ts, bdf->time[0], bdf->work[0], &stageok));
    if (!stageok) goto reject_step;

    bdf->status = TS_STEP_PENDING;
    PetscCall(TSAdaptCandidatesClear(ts->adapt));
    PetscCall(TSAdaptCandidateAdd(ts->adapt, BDF_SchemeName[bdf->k], bdf->k, 1, 1.0, 1.0, PETSC_TRUE));
    PetscCall(TSAdaptChoose(ts->adapt, ts, ts->time_step, NULL, &next_time_step, &accept));
    bdf->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      ts->time_step = next_time_step;
      goto reject_step;
    }

    PetscCall(VecCopy(bdf->work[0], ts->vec_sol));
    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++;
    accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n", ts->steps, rejections));
      ts->reason = TS_DIVERGED_STEP_REJECTED;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSInterpolate_BDF(TS ts, PetscReal t, Vec X)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;

  PetscFunctionBegin;
  PetscCall(TSBDF_Interpolate(ts, bdf->k, t, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSEvaluateWLTE_BDF(TS ts, NormType wnormtype, PetscInt *order, PetscReal *wlte)
{
  TS_BDF   *bdf = (TS_BDF *)ts->data;
  PetscInt  k   = bdf->k;
  PetscReal wltea, wlter;
  Vec       X = bdf->work[0], Y = bdf->vec_lte;

  PetscFunctionBegin;
  k = PetscMin(k, bdf->n - 1);
  PetscCall(TSBDF_VecLTE(ts, k, Y));
  PetscCall(VecAXPY(Y, 1, X));
  PetscCall(TSErrorWeightedNorm(ts, X, Y, wnormtype, wlte, &wltea, &wlter));
  if (order) *order = k + 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSRollBack_BDF(TS ts)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecCopy(bdf->work[1], ts->vec_sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTSFormFunction_BDF(SNES snes, Vec X, Vec F, TS ts)
{
  TS_BDF   *bdf = (TS_BDF *)ts->data;
  DM        dm, dmsave = ts->dm;
  PetscReal t     = bdf->time[0];
  PetscReal shift = bdf->shift;
  Vec       V, V0;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(TSBDF_GetVecs(ts, dm, &V, &V0));
  if (bdf->transientvar) { /* shift*C(X) + V0 */
    PetscCall(TSComputeTransientVariable(ts, X, V));
    PetscCall(VecAYPX(V, shift, V0));
  } else { /* shift*X + V0 */
    PetscCall(VecWAXPY(V, shift, X, V0));
  }

  /* F = Function(t,X,V) */
  ts->dm = dm;
  PetscCall(TSComputeIFunction(ts, t, X, V, F, PETSC_FALSE));
  ts->dm = dmsave;

  PetscCall(TSBDF_RestoreVecs(ts, dm, &V, &V0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTSFormJacobian_BDF(SNES snes, Vec X, Mat J, Mat P, TS ts)
{
  TS_BDF   *bdf = (TS_BDF *)ts->data;
  DM        dm, dmsave = ts->dm;
  PetscReal t     = bdf->time[0];
  PetscReal shift = bdf->shift;
  Vec       V, V0;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(TSBDF_GetVecs(ts, dm, &V, &V0));

  /* J,P = Jacobian(t,X,V) */
  ts->dm = dm;
  PetscCall(TSComputeIJacobian(ts, t, X, V, shift, J, P, PETSC_FALSE));
  ts->dm = dmsave;

  PetscCall(TSBDF_RestoreVecs(ts, dm, &V, &V0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_BDF(TS ts)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;
  size_t  i, n = sizeof(bdf->work) / sizeof(Vec);

  PetscFunctionBegin;
  bdf->k = bdf->n = 0;
  for (i = 0; i < n; i++) {
    PetscCall(VecDestroy(&bdf->work[i]));
    PetscCall(VecDestroy(&bdf->tvwork[i]));
  }
  PetscCall(VecDestroy(&bdf->vec_dot));
  PetscCall(VecDestroy(&bdf->vec_wrk));
  PetscCall(VecDestroy(&bdf->vec_lte));
  if (ts->dm) PetscCall(DMCoarsenHookRemove(ts->dm, DMCoarsenHook_TSBDF, DMRestrictHook_TSBDF, ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_BDF(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_BDF(ts));
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBDFSetOrder_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBDFGetOrder_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_BDF(TS ts)
{
  TS_BDF   *bdf = (TS_BDF *)ts->data;
  size_t    i, n = sizeof(bdf->work) / sizeof(Vec);
  PetscReal low, high, two = 2;

  PetscFunctionBegin;
  PetscCall(TSHasTransientVariable(ts, &bdf->transientvar));
  bdf->k = bdf->n = 0;
  for (i = 0; i < n; i++) {
    PetscCall(VecDuplicate(ts->vec_sol, &bdf->work[i]));
    if (i && bdf->transientvar) PetscCall(VecDuplicate(ts->vec_sol, &bdf->tvwork[i]));
  }
  PetscCall(VecDuplicate(ts->vec_sol, &bdf->vec_dot));
  PetscCall(VecDuplicate(ts->vec_sol, &bdf->vec_wrk));
  PetscCall(VecDuplicate(ts->vec_sol, &bdf->vec_lte));
  PetscCall(TSGetDM(ts, &ts->dm));
  PetscCall(DMCoarsenHookAdd(ts->dm, DMCoarsenHook_TSBDF, DMRestrictHook_TSBDF, ts));

  PetscCall(TSGetAdapt(ts, &ts->adapt));
  PetscCall(TSAdaptCandidatesClear(ts->adapt));
  PetscCall(TSAdaptGetClip(ts->adapt, &low, &high));
  PetscCall(TSAdaptSetClip(ts->adapt, low, PetscMin(high, two)));

  PetscCall(TSGetSNES(ts, &ts->snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_BDF(TS ts, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "BDF ODE solver options");
  {
    PetscBool flg;
    PetscInt  order;
    PetscCall(TSBDFGetOrder(ts, &order));
    PetscCall(PetscOptionsInt("-ts_bdf_order", "Order of the BDF method", "TSBDFSetOrder", order, &order, &flg));
    if (flg) PetscCall(TSBDFSetOrder(ts, order));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_BDF(TS ts, PetscViewer viewer)
{
  TS_BDF   *bdf = (TS_BDF *)ts->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  Order=%" PetscInt_FMT "\n", bdf->order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------ */

static PetscErrorCode TSBDFSetOrder_BDF(TS ts, PetscInt order)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;

  PetscFunctionBegin;
  if (order == bdf->order) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(order >= 1 && order <= 6, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "BDF Order %" PetscInt_FMT " not implemented", order);
  bdf->order = order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSBDFGetOrder_BDF(TS ts, PetscInt *order)
{
  TS_BDF *bdf = (TS_BDF *)ts->data;

  PetscFunctionBegin;
  *order = bdf->order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------ */

/*MC
      TSBDF - DAE solver using BDF methods

  Level: beginner

.seealso: [](chapter_ts), `TS`, `TSCreate()`, `TSSetType()`, `TSType`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_BDF(TS ts)
{
  TS_BDF *bdf;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_BDF;
  ts->ops->destroy        = TSDestroy_BDF;
  ts->ops->view           = TSView_BDF;
  ts->ops->setup          = TSSetUp_BDF;
  ts->ops->setfromoptions = TSSetFromOptions_BDF;
  ts->ops->step           = TSStep_BDF;
  ts->ops->evaluatewlte   = TSEvaluateWLTE_BDF;
  ts->ops->rollback       = TSRollBack_BDF;
  ts->ops->interpolate    = TSInterpolate_BDF;
  ts->ops->snesfunction   = SNESTSFormFunction_BDF;
  ts->ops->snesjacobian   = SNESTSFormJacobian_BDF;
  ts->default_adapt_type  = TSADAPTBASIC;

  ts->usessnes = PETSC_TRUE;

  PetscCall(PetscNew(&bdf));
  ts->data = (void *)bdf;

  bdf->status = TS_STEP_COMPLETE;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBDFSetOrder_C", TSBDFSetOrder_BDF));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSBDFGetOrder_C", TSBDFGetOrder_BDF));
  PetscCall(TSBDFSetOrder(ts, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------ */

/*@
  TSBDFSetOrder - Set the order of the `TSBDF` method

  Logically Collective

  Input Parameters:
+  ts - timestepping context
-  order - order of the method

  Options Database Key:
.  -ts_bdf_order <order> - select the order

  Level: intermediate

.seealso: `TSBDFGetOrder()`, `TS`, `TSBDF`
@*/
PetscErrorCode TSBDFSetOrder(TS ts, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ts, order, 2);
  PetscTryMethod(ts, "TSBDFSetOrder_C", (TS, PetscInt), (ts, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSBDFGetOrder - Get the order of the `TSBDF` method

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  order - order of the method

  Level: intermediate

.seealso: `TSBDFSetOrder()`, `TS`, `TSBDF`
@*/
PetscErrorCode TSBDFGetOrder(TS ts, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidIntPointer(order, 2);
  PetscUseMethod(ts, "TSBDFGetOrder_C", (TS, PetscInt *), (ts, order));
  PetscFunctionReturn(PETSC_SUCCESS);
}
