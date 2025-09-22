#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petscdm.h>
#include <../src/ts/impls/arkimex/arkimex.h>
#include <../src/ts/impls/arkimex/fsarkimex.h>

static PetscErrorCode TSARKIMEXSetSplits(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  DM          dm, subdm, newdm;

  PetscFunctionBegin;
  PetscCall(TSRHSSplitGetSubTS(ts, "slow", &ark->subts_slow));
  PetscCall(TSRHSSplitGetSubTS(ts, "fast", &ark->subts_fast));
  /* Only copy the DM */
  PetscCall(TSGetDM(ts, &dm));
  if (ark->subts_slow) {
    PetscCall(DMClone(dm, &newdm));
    PetscCall(TSGetDM(ark->subts_slow, &subdm));
    PetscCall(DMCopyDMTS(subdm, newdm));
    PetscCall(TSSetDM(ark->subts_slow, newdm));
    PetscCall(DMDestroy(&newdm));
  }
  if (ark->subts_fast) {
    PetscCall(DMClone(dm, &newdm));
    PetscCall(TSGetDM(ark->subts_fast, &subdm));
    PetscCall(DMCopyDMTS(subdm, newdm));
    PetscCall(TSSetDM(ark->subts_fast, newdm));
    PetscCall(DMDestroy(&newdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTSFormFunction_ARKIMEX_FastSlowSplit(SNES snes, Vec X, Vec F, TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  DM          dm, dmsave;
  Vec         Z = ark->Z, Ydot = ark->Ydot, Y = ark->Y_snes;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  dmsave = ts->dm;
  ts->dm = dm; // Use the SNES DM to compute IFunction

  PetscReal shift = ark->scoeff / ts->time_step;
  PetscCall(VecAXPBYPCZ(Ydot, -shift, shift, 0, Z, X)); /* Ydot = shift*(X-Z) */
  if (ark->is_slow) PetscCall(VecISCopy(Y, ark->is_fast, SCATTER_FORWARD, X));
  else Y = Z;
  PetscCall(TSComputeIFunction(ark->subts_fast, ark->stage_time, Y, Ydot, F, ark->imex));

  ts->dm = dmsave;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTSFormJacobian_ARKIMEX_FastSlowSplit(SNES snes, Vec X, Mat A, Mat B, TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  DM          dm, dmsave;
  Vec         Z = ark->Z, Ydot = ark->Ydot, Y = ark->Y_snes;
  PetscReal   shift;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  dmsave = ts->dm;
  ts->dm = dm;

  shift = ark->scoeff / ts->time_step;
  if (ark->is_slow) PetscCall(VecISCopy(Y, ark->is_fast, SCATTER_FORWARD, X));
  else Y = Z;
  PetscCall(TSComputeIJacobian(ark->subts_fast, ark->stage_time, Y, Ydot, shift, A, B, ark->imex));

  ts->dm = dmsave;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSExtrapolate_ARKIMEX_FastSlowSplit(TS ts, PetscReal c, Vec X)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau       tab = ark->tableau;
  PetscInt         s = tab->s, pinterp = tab->pinterp, i, j;
  PetscReal        h, h_prev, t, tt;
  PetscScalar     *bt = ark->work, *b = ark->work + s;
  const PetscReal *Bt = tab->binterpt, *B = tab->binterp;
  PetscBool        fasthasE;

  PetscFunctionBegin;
  PetscCheck(Bt && B, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "TSARKIMEX %s does not have an interpolation formula", ark->tableau->name);
  h      = ts->time_step;
  h_prev = ts->ptime - ts->ptime_prev;
  t      = 1 + h / h_prev * c;
  for (i = 0; i < s; i++) bt[i] = b[i] = 0;
  for (j = 0, tt = t; j < pinterp; j++, tt *= t) {
    for (i = 0; i < s; i++) {
      bt[i] += h * Bt[i * pinterp + j] * tt;
      b[i] += h * B[i * pinterp + j] * tt;
    }
  }
  PetscCheck(ark->Y_prev, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Stages from previous step have not been stored");
  PetscCall(VecCopy(ark->Y_prev[0], X));
  PetscCall(VecMAXPY(X, s, bt, ark->YdotI_prev));
  PetscCall(TSHasRHSFunction(ark->subts_fast, &fasthasE));
  if (fasthasE) PetscCall(VecMAXPY(X, s, b, ark->YdotRHS_prev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 The step completion formula is

 x1 = x0 - h bt^T YdotI + h b^T YdotRHS

 This function can be called before or after ts->vec_sol has been updated.
 Suppose we have a completion formula (bt,b) and an embedded formula (bet,be) of different order.
 We can write

 x1e = x0 - h bet^T YdotI + h be^T YdotRHS
     = x1 + h bt^T YdotI - h b^T YdotRHS - h bet^T YdotI + h be^T YdotRHS
     = x1 - h (bet - bt)^T YdotI + h (be - b)^T YdotRHS

 so we can evaluate the method with different order even after the step has been optimistically completed.
*/
static PetscErrorCode TSEvaluateStep_ARKIMEX_FastSlowSplit(TS ts, PetscInt order, Vec X, PetscBool *done)
{
  TS_ARKIMEX  *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau   tab = ark->tableau;
  Vec          Xfast, Xslow;
  PetscScalar *w = ark->work;
  PetscReal    h;
  PetscInt     s = tab->s, j;
  PetscBool    fasthasE;

  PetscFunctionBegin;
  switch (ark->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step;
    break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Invalid TSStepStatus");
  }
  if (ark->is_fast) PetscCall(TSHasRHSFunction(ark->subts_fast, &fasthasE));
  if (order == tab->order) {
    if (ark->status == TS_STEP_INCOMPLETE) {
      PetscCall(VecCopy(ts->vec_sol, X));
      for (j = 0; j < s; j++) w[j] = h * tab->b[j];
      if (ark->is_slow) {
        PetscCall(VecGetSubVector(X, ark->is_slow, &Xslow));
        PetscCall(VecMAXPY(Xslow, s, w, ark->YdotRHS_slow));
        PetscCall(VecRestoreSubVector(X, ark->is_slow, &Xslow));
      }
      if (ark->is_fast) {
        PetscCall(VecGetSubVector(X, ark->is_fast, &Xfast));
        if (fasthasE) PetscCall(VecMAXPY(Xfast, s, w, ark->YdotRHS_fast));
        for (j = 0; j < s; j++) w[j] = h * tab->bt[j];
        PetscCall(VecMAXPY(Xfast, s, w, ark->YdotI_fast));
        PetscCall(VecRestoreSubVector(X, ark->is_fast, &Xfast));
      }
    } else PetscCall(VecCopy(ts->vec_sol, X));
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (order == tab->order - 1) {
    if (!tab->bembedt) goto unavailable;
    if (ark->status == TS_STEP_INCOMPLETE) { /* Complete with the embedded method (bet,be) */
      PetscCall(VecCopy(ts->vec_sol, X));
      for (j = 0; j < s; j++) w[j] = h * tab->bembed[j];
      if (ark->is_slow) {
        PetscCall(VecGetSubVector(X, ark->is_slow, &Xslow));
        PetscCall(VecMAXPY(Xslow, s, w, ark->YdotRHS_slow));
        PetscCall(VecRestoreSubVector(X, ark->is_slow, &Xslow));
      }
      if (ark->is_fast) {
        PetscCall(VecGetSubVector(X, ark->is_fast, &Xfast));
        if (fasthasE) PetscCall(VecMAXPY(Xfast, s, w, ark->YdotRHS_fast));
        for (j = 0; j < s; j++) w[j] = h * tab->bembedt[j];
        PetscCall(VecMAXPY(Xfast, s, w, ark->YdotI_fast));
        PetscCall(VecRestoreSubVector(X, ark->is_fast, &Xfast));
      }
    } else { /* Rollback and re-complete using (bet-be,be-b) */
      PetscCall(VecCopy(ts->vec_sol, X));
      for (j = 0; j < s; j++) w[j] = h * (tab->bembed[j] - tab->b[j]);
      if (ark->is_slow) {
        PetscCall(VecGetSubVector(X, ark->is_slow, &Xslow));
        PetscCall(VecMAXPY(Xslow, s, w, ark->YdotRHS_slow));
        PetscCall(VecRestoreSubVector(X, ark->is_slow, &Xslow));
      }
      if (ark->is_fast) {
        PetscCall(VecGetSubVector(X, ark->is_fast, &Xfast));
        if (fasthasE) PetscCall(VecMAXPY(Xfast, tab->s, w, ark->YdotRHS_fast));
        for (j = 0; j < s; j++) w[j] = h * (tab->bembedt[j] - tab->bt[j]);
        PetscCall(VecMAXPY(Xfast, tab->s, w, ark->YdotI_fast));
        PetscCall(VecRestoreSubVector(X, ark->is_fast, &Xfast));
      }
    }
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
unavailable:
  PetscCheck(done, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "ARKIMEX '%s' of order %" PetscInt_FMT " cannot evaluate step at order %" PetscInt_FMT ". Consider using -ts_adapt_type none or a different method that has an embedded estimate.",
             tab->name, tab->order, order);
  *done = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Additive Runge-Kutta methods for a fast-slow (component-wise partitioned) system in the form
    Ufdot = Ff(t,Uf,Us)
    Usdot = Fs(t,Uf,Us)

  Ys[i] = Us_n + dt \sum_{j=1}^{i-1} a[i][j] Fs(t+c_j*dt,Yf[j],Ys[j])
  Ys[i] = Us_n + dt \sum_{j=1}^{i-1} a[i][j] Fs(t+c_j*dt,Yf[j],Ys[j])

*/
static PetscErrorCode TSStep_ARKIMEX_FastSlowSplit(TS ts)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau       tab = ark->tableau;
  const PetscInt   s   = tab->s;
  const PetscReal *At = tab->At, *A = tab->A, *ct = tab->ct;
  PetscScalar     *w = ark->work;
  Vec             *Y = ark->Y, Ydot_fast = ark->Ydot, Ydot0_fast = ark->Ydot0, Z = ark->Z, *YdotRHS_fast = ark->YdotRHS_fast, *YdotRHS_slow = ark->YdotRHS_slow, *YdotI_fast = ark->YdotI_fast, Yfast, Yslow, Xfast, Xslow;
  PetscBool        extrapolate = ark->extrapolate;
  TSAdapt          adapt;
  SNES             snes;
  PetscInt         i, j, its, lits;
  PetscInt         rejections = 0;
  PetscBool        fasthasE = PETSC_FALSE, stageok, accept = PETSC_TRUE;
  PetscReal        next_time_step = ts->time_step;

  PetscFunctionBegin;
  if (ark->is_fast) PetscCall(TSHasRHSFunction(ark->subts_fast, &fasthasE));
  if (ark->extrapolate && !ark->Y_prev) {
    PetscCall(VecGetSubVector(ts->vec_sol, ark->is_fast, &Xfast));
    PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->Y_prev));
    PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->YdotI_prev));
    if (fasthasE) PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->YdotRHS_prev));
    PetscCall(VecRestoreSubVector(ts->vec_sol, ark->is_fast, &Xfast));
    PetscCall(VecGetSubVector(ts->vec_sol, ark->is_slow, &Xslow));
    PetscCall(VecRestoreSubVector(ts->vec_sol, ark->is_fast, &Xslow));
  }

  if (!ts->steprollback) {
    if (ts->equation_type >= TS_EQ_IMPLICIT) { /* Save the initial slope for the next step */
      PetscCall(VecCopy(YdotI_fast[s - 1], Ydot0_fast));
    }
    if (ark->extrapolate && !ts->steprestart) { /* Save the Y, YdotI, YdotRHS for extrapolation initial guess */
      for (i = 0; i < s; i++) {
        PetscCall(VecISCopy(Y[i], ark->is_fast, SCATTER_REVERSE, ark->Y_prev[i]));
        PetscCall(VecCopy(YdotI_fast[i], ark->YdotI_prev[i]));
        if (fasthasE) PetscCall(VecCopy(YdotRHS_fast[i], ark->YdotRHS_prev[i]));
      }
    }
  }

  /* For IMEX we compute a step */
  if (ts->equation_type >= TS_EQ_IMPLICIT && tab->explicit_first_stage && ts->steprestart) {
    TS ts_start;
    PetscCall(TSClone(ts, &ts_start));
    PetscCall(TSSetSolution(ts_start, ts->vec_sol));
    PetscCall(TSSetTime(ts_start, ts->ptime));
    PetscCall(TSSetMaxSteps(ts_start, ts->steps + 1));
    PetscCall(TSSetMaxTime(ts_start, ts->ptime + ts->time_step));
    PetscCall(TSSetExactFinalTime(ts_start, TS_EXACTFINALTIME_STEPOVER));
    PetscCall(TSSetTimeStep(ts_start, ts->time_step));
    PetscCall(TSSetType(ts_start, TSARKIMEX));
    PetscCall(TSARKIMEXSetFullyImplicit(ts_start, PETSC_TRUE));
    PetscCall(TSARKIMEXSetType(ts_start, TSARKIMEX1BEE));

    PetscCall(TSRestartStep(ts_start));
    PetscCall(TSSolve(ts_start, ts->vec_sol));
    PetscCall(TSGetTime(ts_start, &ts->ptime));
    PetscCall(TSGetTimeStep(ts_start, &ts->time_step));

    { /* Save the initial slope for the next step */
      TS_ARKIMEX *ark_start = (TS_ARKIMEX *)ts_start->data;
      PetscCall(VecCopy(ark_start->YdotI[ark_start->tableau->s - 1], Ydot0_fast));
    }
    ts->steps++;
    /* Set the correct TS in SNES */
    /* We'll try to bypass this by changing the method on the fly */
    {
      PetscCall(TSRHSSplitGetSNES(ts, &snes));
      PetscCall(TSRHSSplitSetSNES(ts, snes));
    }
    PetscCall(TSDestroy(&ts_start));
  }

  ark->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && ark->status != TS_STEP_COMPLETE) {
    PetscReal t = ts->ptime;
    PetscReal h = ts->time_step;
    for (i = 0; i < s; i++) {
      ark->stage_time = t + h * ct[i];
      PetscCall(TSPreStage(ts, ark->stage_time));
      PetscCall(VecCopy(ts->vec_sol, Y[i]));
      /* fast components */
      if (ark->is_fast) {
        if (At[i * s + i] == 0) { /* This stage is explicit */
          PetscCheck(i == 0 || ts->equation_type < TS_EQ_IMPLICIT, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Explicit stages other than the first one are not supported for implicit problems");
          PetscCall(VecGetSubVector(Y[i], ark->is_fast, &Yfast));
          for (j = 0; j < i; j++) w[j] = h * At[i * s + j];
          PetscCall(VecMAXPY(Yfast, i, w, YdotI_fast));
          if (fasthasE) {
            for (j = 0; j < i; j++) w[j] = h * A[i * s + j];
            PetscCall(VecMAXPY(Yfast, i, w, YdotRHS_fast));
          }
          PetscCall(VecRestoreSubVector(Y[i], ark->is_fast, &Yfast));
        } else {
          ark->scoeff = 1. / At[i * s + i];
          /* Ydot = shift*(Y-Z) */
          PetscCall(VecISCopy(ts->vec_sol, ark->is_fast, SCATTER_REVERSE, Z));
          for (j = 0; j < i; j++) w[j] = h * At[i * s + j];
          PetscCall(VecMAXPY(Z, i, w, YdotI_fast));
          if (fasthasE) {
            for (j = 0; j < i; j++) w[j] = h * A[i * s + j];
            PetscCall(VecMAXPY(Z, i, w, YdotRHS_fast));
          }
          PetscCall(TSRHSSplitGetSNES(ts, &snes));
          if (ark->is_slow) PetscCall(VecCopy(i > 0 ? Y[i - 1] : ts->vec_sol, ark->Y_snes));
          else ark->Y_snes = Y[i];
          PetscCall(VecGetSubVector(Y[i], ark->is_fast, &Yfast));
          if (extrapolate && !ts->steprestart) {
            /* Initial guess extrapolated from previous time step stage values */
            PetscCall(TSExtrapolate_ARKIMEX_FastSlowSplit(ts, ct[i], Yfast));
          } else {
            /* Initial guess taken from last stage */
            PetscCall(VecISCopy(i > 0 ? Y[i - 1] : ts->vec_sol, ark->is_fast, SCATTER_REVERSE, Yfast));
          }
          PetscCall(SNESSolve(snes, NULL, Yfast));
          PetscCall(VecRestoreSubVector(Y[i], ark->is_fast, &Yfast));
          PetscCall(SNESGetIterationNumber(snes, &its));
          PetscCall(SNESGetLinearSolveIterations(snes, &lits));
          ts->snes_its += its;
          ts->ksp_its += lits;
          PetscCall(TSGetAdapt(ts, &adapt));
          PetscCall(TSAdaptCheckStage(adapt, ts, ark->stage_time, Y[i], &stageok));
          if (!stageok) {
            /* We are likely rejecting the step because of solver or function domain problems so we should not attempt to
             * use extrapolation to initialize the solves on the next attempt. */
            extrapolate = PETSC_FALSE;
            goto reject_step;
          }
        }

        if (ts->equation_type >= TS_EQ_IMPLICIT) {
          if (i == 0 && tab->explicit_first_stage) {
            PetscCheck(tab->stiffly_accurate, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "%s %s is not stiffly accurate and therefore explicit-first stage methods cannot be used if the equation is implicit because the slope cannot be evaluated",
                       ((PetscObject)ts)->type_name, ark->tableau->name);
            PetscCall(VecCopy(Ydot0_fast, YdotI_fast[0])); /* YdotI_fast = YdotI_fast(tn-1) */
          } else {
            PetscCall(VecGetSubVector(Y[i], ark->is_fast, &Yfast));
            PetscCall(VecAXPBYPCZ(YdotI_fast[i], -ark->scoeff / h, ark->scoeff / h, 0, Z, Yfast)); /* YdotI = shift*(X-Z) */
            PetscCall(VecRestoreSubVector(Y[i], ark->is_fast, &Yfast));
          }
        } else {
          if (i == 0 && tab->explicit_first_stage) {
            PetscCall(VecZeroEntries(Ydot_fast));
            PetscCall(TSComputeIFunction(ark->subts_fast, ark->stage_time, Y[i], Ydot_fast, YdotI_fast[i], ark->imex)); /* YdotI = -G(t,Y,0)   */
            PetscCall(VecScale(YdotI_fast[i], -1.0));
          } else {
            PetscCall(VecGetSubVector(Y[i], ark->is_fast, &Yfast));
            PetscCall(VecAXPBYPCZ(YdotI_fast[i], -ark->scoeff / h, ark->scoeff / h, 0, Z, Yfast)); /* YdotI = shift*(X-Z) */
            PetscCall(VecRestoreSubVector(Y[i], ark->is_fast, &Yfast));
          }
          if (fasthasE) PetscCall(TSComputeRHSFunction(ark->subts_fast, ark->stage_time, Y[i], YdotRHS_fast[i]));
        }
      }
      /* slow components */
      if (ark->is_slow) {
        for (j = 0; j < i; j++) w[j] = h * A[i * s + j];
        PetscCall(VecGetSubVector(Y[i], ark->is_slow, &Yslow));
        PetscCall(VecMAXPY(Yslow, i, w, YdotRHS_slow));
        PetscCall(VecRestoreSubVector(Y[i], ark->is_slow, &Yslow));
        PetscCall(TSComputeRHSFunction(ark->subts_slow, ark->stage_time, Y[i], YdotRHS_slow[i]));
      }
      PetscCall(TSPostStage(ts, ark->stage_time, i, Y));
    }
    ark->status = TS_STEP_INCOMPLETE;
    PetscCall(TSEvaluateStep_ARKIMEX_FastSlowSplit(ts, tab->order, ts->vec_sol, NULL));
    ark->status = TS_STEP_PENDING;
    PetscCall(TSGetAdapt(ts, &adapt));
    PetscCall(TSAdaptCandidatesClear(adapt));
    PetscCall(TSAdaptCandidateAdd(adapt, tab->name, tab->order, 1, tab->ccfl, (PetscReal)tab->s, PETSC_TRUE));
    PetscCall(TSAdaptChoose(adapt, ts, ts->time_step, NULL, &next_time_step, &accept));
    ark->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) { /* Roll back the current step */
      PetscCall(VecCopy(ts->vec_sol0, ts->vec_sol));
      ts->time_step = next_time_step;
      goto reject_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++;
    accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n", ts->steps, rejections));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_ARKIMEX_FastSlowSplit(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau  tab = ark->tableau;
  Vec         Xfast, Xslow;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(2 * tab->s, &ark->work));
  PetscCall(VecDuplicateVecs(ts->vec_sol, tab->s, &ark->Y));
  PetscCall(TSRHSSplitGetIS(ts, "slow", &ark->is_slow));
  PetscCall(TSRHSSplitGetIS(ts, "fast", &ark->is_fast));
  PetscCheck(ark->is_slow || ark->is_fast, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set up RHSSplits with TSRHSSplitSetIS() using split names 'slow' or 'fast' or both in order to use -ts_arkimex_fastslow true");
  /* The following vectors need to be resized */
  PetscCall(VecDestroy(&ark->Ydot));
  PetscCall(VecDestroy(&ark->Ydot0));
  PetscCall(VecDestroy(&ark->Z));
  PetscCall(VecDestroyVecs(tab->s, &ark->YdotI_fast));
  if (ark->extrapolate && ark->is_slow) { // need to resize these vectors if the fast subvectors is smaller than their original counterparts (which means IS)
    PetscCall(VecDestroyVecs(tab->s, &ark->Y_prev));
    PetscCall(VecDestroyVecs(tab->s, &ark->YdotI_prev));
    PetscCall(VecDestroyVecs(tab->s, &ark->YdotRHS_prev));
  }
  /* Allocate working vectors */
  if (ark->is_fast && ark->is_slow) PetscCall(VecDuplicate(ts->vec_sol, &ark->Y_snes)); // need an additional work vector for SNES
  if (ark->is_fast) {
    PetscCall(VecGetSubVector(ts->vec_sol, ark->is_fast, &Xfast));
    PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->YdotRHS_fast));
    PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->YdotI_fast));
    PetscCall(VecDuplicate(Xfast, &ark->Ydot));
    PetscCall(VecDuplicate(Xfast, &ark->Ydot0));
    PetscCall(VecDuplicate(Xfast, &ark->Z));
    if (ark->extrapolate) {
      PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->Y_prev));
      PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->YdotI_prev));
      PetscCall(VecDuplicateVecs(Xfast, tab->s, &ark->YdotRHS_prev));
    }
    PetscCall(VecRestoreSubVector(ts->vec_sol, ark->is_fast, &Xfast));
  }
  if (ark->is_slow) {
    PetscCall(VecGetSubVector(ts->vec_sol, ark->is_slow, &Xslow));
    PetscCall(VecDuplicateVecs(Xslow, tab->s, &ark->YdotRHS_slow));
    PetscCall(VecRestoreSubVector(ts->vec_sol, ark->is_slow, &Xslow));
  }
  ts->ops->step         = TSStep_ARKIMEX_FastSlowSplit;
  ts->ops->evaluatestep = TSEvaluateStep_ARKIMEX_FastSlowSplit;
  PetscCall(TSARKIMEXSetSplits(ts));
  if (ark->subts_fast) { // subts SNESJacobian is set when users set the subts Jacobian, but the main ts SNESJacobian needs to be set too
    SNES snes, snes_fast;
    Mat  Amat, Pmat;
    PetscErrorCode (*func)(SNES, Vec, Mat, Mat, void *);
    PetscCall(TSRHSSplitGetSNES(ts, &snes));
    PetscCall(TSGetSNES(ark->subts_fast, &snes_fast));
    PetscCall(SNESGetJacobian(snes_fast, &Amat, &Pmat, &func, NULL));
    if (func == SNESTSFormJacobian) PetscCall(SNESSetJacobian(snes, Amat, Pmat, SNESTSFormJacobian, ts));
    ts->ops->snesfunction = SNESTSFormFunction_ARKIMEX_FastSlowSplit;
    ts->ops->snesjacobian = SNESTSFormJacobian_ARKIMEX_FastSlowSplit;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_ARKIMEX_FastSlowSplit(TS ts)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;
  ARKTableau  tab = ark->tableau;

  PetscFunctionBegin;
  if (tab) {
    PetscCall(PetscFree(ark->work));
    PetscCall(VecDestroyVecs(tab->s, &ark->Y));
    if (ark->is_fast && ark->is_slow) PetscCall(VecDestroy(&ark->Y_snes));
    PetscCall(VecDestroyVecs(tab->s, &ark->YdotRHS_slow));
    PetscCall(VecDestroyVecs(tab->s, &ark->YdotRHS_fast));
    PetscCall(VecDestroyVecs(tab->s, &ark->YdotI_fast));
    PetscCall(VecDestroy(&ark->Ydot));
    PetscCall(VecDestroy(&ark->Ydot0));
    PetscCall(VecDestroy(&ark->Z));
    if (ark->extrapolate) {
      PetscCall(VecDestroyVecs(tab->s, &ark->Y_prev));
      PetscCall(VecDestroyVecs(tab->s, &ark->YdotI_prev));
      PetscCall(VecDestroyVecs(tab->s, &ark->YdotRHS_prev));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSARKIMEXSetFastSlowSplit_ARKIMEX(TS ts, PetscBool fastslowsplit)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  ark->fastslowsplit = fastslowsplit;
  if (fastslowsplit) {
    PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSetUp_ARKIMEX_FastSlowSplit_C", TSSetUp_ARKIMEX_FastSlowSplit));
    PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSReset_ARKIMEX_FastSlowSplit_C", TSReset_ARKIMEX_FastSlowSplit));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSARKIMEXGetFastSlowSplit_ARKIMEX(TS ts, PetscBool *fastslowsplit)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX *)ts->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  *fastslowsplit = ark->fastslowsplit;
  PetscFunctionReturn(PETSC_SUCCESS);
}
