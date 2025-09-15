/*
       Code for Timestepping with implicit backwards Euler.
*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/

typedef struct {
  Vec update; /* work vector where new solution is formed */
  Vec func;   /* work vector where F(t[i],u[i]) is stored */
  Vec xdot;   /* work vector for time derivative of state */

  /* information used for Pseudo-timestepping */

  PetscErrorCode (*dt)(TS, PetscReal *, void *); /* compute next timestep, and related context */
  void *dtctx;
  PetscErrorCode (*verify)(TS, Vec, void *, PetscReal *, PetscBool *); /* verify previous timestep and related context */
  void *verifyctx;

  PetscReal fnorm_initial, fnorm; /* original and current norm of F(u) */
  PetscReal fnorm_previous;

  PetscReal dt_initial;   /* initial time-step */
  PetscReal dt_increment; /* scaling that dt is incremented each time-step */
  PetscReal dt_max;       /* maximum time step */
  PetscBool increment_dt_from_initial_dt;
  PetscReal fatol, frtol;

  PetscObjectState Xstate; /* state of input vector to TSComputeIFunction() with 0 Xdot */
} TS_Pseudo;

/* ------------------------------------------------------------------------------*/

/*@
  TSPseudoComputeTimeStep - Computes the next timestep for a currently running
  pseudo-timestepping process.

  Collective

  Input Parameter:
. ts - timestep context

  Output Parameter:
. dt - newly computed timestep

  Level: developer

  Note:
  The routine to be called here to compute the timestep should be
  set by calling `TSPseudoSetTimeStep()`.

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoTimeStepDefault()`, `TSPseudoSetTimeStep()`
@*/
PetscErrorCode TSPseudoComputeTimeStep(TS ts, PetscReal *dt)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(TS_PseudoComputeTimeStep, ts, 0, 0, 0));
  PetscCall((*pseudo->dt)(ts, dt, pseudo->dtctx));
  PetscCall(PetscLogEventEnd(TS_PseudoComputeTimeStep, ts, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------------------*/
/*@C
  TSPseudoVerifyTimeStepDefault - Default code to verify the quality of the last timestep.

  Collective, No Fortran Support

  Input Parameters:
+ ts     - the timestep context
. dtctx  - unused timestep context
- update - latest solution vector

  Output Parameters:
+ newdt - the timestep to use for the next step
- flag  - flag indicating whether the last time step was acceptable

  Level: advanced

  Note:
  This routine always returns a flag of 1, indicating an acceptable
  timestep.

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoSetVerifyTimeStep()`, `TSPseudoVerifyTimeStep()`
@*/
PetscErrorCode TSPseudoVerifyTimeStepDefault(TS ts, Vec update, void *dtctx, PetscReal *newdt, PetscBool *flag)
{
  PetscFunctionBegin;
  *flag = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSPseudoVerifyTimeStep - Verifies whether the last timestep was acceptable.

  Collective

  Input Parameters:
+ ts     - timestep context
- update - latest solution vector

  Output Parameters:
+ dt   - newly computed timestep (if it had to shrink)
- flag - indicates if current timestep was ok

  Level: advanced

  Notes:
  The routine to be called here to compute the timestep should be
  set by calling `TSPseudoSetVerifyTimeStep()`.

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoSetVerifyTimeStep()`, `TSPseudoVerifyTimeStepDefault()`
@*/
PetscErrorCode TSPseudoVerifyTimeStep(TS ts, Vec update, PetscReal *dt, PetscBool *flag)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  *flag = PETSC_TRUE;
  if (pseudo->verify) PetscCall((*pseudo->verify)(ts, update, pseudo->verifyctx, dt, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------------------------------------------------------*/

static PetscErrorCode TSStep_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;
  PetscInt   nits, lits, reject;
  PetscBool  stepok;
  PetscReal  next_time_step = ts->time_step;

  PetscFunctionBegin;
  if (ts->steps == 0) {
    pseudo->dt_initial = ts->time_step;
    PetscCall(VecCopy(ts->vec_sol, pseudo->update)); /* in all future updates pseudo->update already contains the current time solution */
  }
  PetscCall(TSPseudoComputeTimeStep(ts, &next_time_step));
  for (reject = 0; reject < ts->max_reject; reject++, ts->reject++) {
    ts->time_step = next_time_step;
    if (reject) PetscCall(VecCopy(ts->vec_sol, pseudo->update)); /* need to copy the solution because we got a rejection and pseudo->update contains intermediate computation */
    PetscCall(TSPreStage(ts, ts->ptime + ts->time_step));
    PetscCall(SNESSolve(ts->snes, NULL, pseudo->update));
    PetscCall(SNESGetIterationNumber(ts->snes, &nits));
    PetscCall(SNESGetLinearSolveIterations(ts->snes, &lits));
    ts->snes_its += nits;
    ts->ksp_its += lits;
    PetscCall(TSPostStage(ts, ts->ptime + ts->time_step, 0, &pseudo->update));
    PetscCall(TSAdaptCheckStage(ts->adapt, ts, ts->ptime + ts->time_step, pseudo->update, &stepok));
    if (!stepok) {
      next_time_step = ts->time_step;
      continue;
    }
    pseudo->fnorm = -1; /* The current norm is no longer valid */
    PetscCall(TSPseudoVerifyTimeStep(ts, pseudo->update, &next_time_step, &stepok));
    if (stepok) break;
  }
  if (reject >= ts->max_reject) {
    ts->reason = TS_DIVERGED_STEP_REJECTED;
    PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n", ts->steps, reject));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecCopy(pseudo->update, ts->vec_sol));
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;

  PetscCall(VecZeroEntries(pseudo->xdot));
  PetscCall(TSComputeIFunction(ts, ts->ptime, pseudo->update, pseudo->xdot, pseudo->func, PETSC_FALSE));
  PetscCall(PetscObjectStateGet((PetscObject)pseudo->update, &pseudo->Xstate));
  PetscCall(VecNorm(pseudo->func, NORM_2, &pseudo->fnorm));

  if (pseudo->fnorm < pseudo->fatol) {
    ts->reason = TS_CONVERGED_PSEUDO_FATOL;
    PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", converged since fnorm %g < fatol %g\n", ts->steps, (double)pseudo->fnorm, (double)pseudo->frtol));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (pseudo->fnorm / pseudo->fnorm_initial < pseudo->frtol) {
    ts->reason = TS_CONVERGED_PSEUDO_FRTOL;
    PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", converged since fnorm %g / fnorm_initial %g < frtol %g\n", ts->steps, (double)pseudo->fnorm, (double)pseudo->fnorm_initial, (double)pseudo->fatol));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSReset_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&pseudo->update));
  PetscCall(VecDestroy(&pseudo->func));
  PetscCall(VecDestroy(&pseudo->xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_Pseudo(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_Pseudo(ts));
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetVerifyTimeStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetTimeStepIncrement_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetMaxTimeStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoIncrementDtFromInitialDt_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetTimeStep_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSPseudoGetXdot(TS ts, Vec X, Vec *Xdot)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  *Xdot = pseudo->xdot;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    The transient residual is

        F(U^{n+1},(U^{n+1}-U^n)/dt) = 0

    or for ODE,

        (U^{n+1} - U^{n})/dt - F(U^{n+1}) = 0

    This is the function that must be evaluated for transient simulation and for
    finite difference Jacobians.  On the first Newton step, this algorithm uses
    a guess of U^{n+1} = U^n in which case the transient term vanishes and the
    residual is actually the steady state residual.  Pseudotransient
    continuation as described in the literature is a linearly implicit
    algorithm, it only takes this one full Newton step with the steady state
    residual, and then advances to the next time step.

    This routine avoids a repeated identical call to TSComputeRHSFunction() when that result
    is already contained in pseudo->func due to the call to TSComputeIFunction() in TSStep_Pseudo()

*/
static PetscErrorCode SNESTSFormFunction_Pseudo(SNES snes, Vec X, Vec Y, TS ts)
{
  TSIFunctionFn    *ifunction;
  TSRHSFunctionFn  *rhsfunction;
  void             *ctx;
  DM                dm;
  TS_Pseudo        *pseudo = (TS_Pseudo *)ts->data;
  const PetscScalar mdt    = 1.0 / ts->time_step;
  PetscBool         KSPSNES;
  PetscObjectState  Xstate;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 4);

  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMTSGetIFunction(dm, &ifunction, &ctx));
  PetscCall(DMTSGetRHSFunction(dm, &rhsfunction, NULL));
  PetscCheck(rhsfunction || ifunction, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must call TSSetRHSFunction() and / or TSSetIFunction()");

  PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESKSPONLY, &KSPSNES));
  PetscCall(PetscObjectStateGet((PetscObject)X, &Xstate));
  if (Xstate != pseudo->Xstate || ifunction || !KSPSNES) {
    PetscCall(VecAXPBYPCZ(pseudo->xdot, -mdt, mdt, 0, ts->vec_sol, X));
    PetscCall(TSComputeIFunction(ts, ts->ptime + ts->time_step, X, pseudo->xdot, Y, PETSC_FALSE));
  } else {
    /* reuse the TSComputeIFunction() result performed inside TSStep_Pseudo() */
    /* note that pseudo->func contains the negation of TSComputeRHSFunction() */
    PetscCall(VecWAXPY(Y, 1, pseudo->func, pseudo->xdot));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   This constructs the Jacobian needed for SNES.  For DAE, this is

       dF(X,Xdot)/dX + shift*dF(X,Xdot)/dXdot

    and for ODE:

       J = I/dt - J_{Frhs}   where J_{Frhs} is the given Jacobian of Frhs.
*/
static PetscErrorCode SNESTSFormJacobian_Pseudo(SNES snes, Vec X, Mat AA, Mat BB, TS ts)
{
  Vec Xdot;

  PetscFunctionBegin;
  /* Xdot has already been computed in SNESTSFormFunction_Pseudo (SNES guarantees this) */
  PetscCall(TSPseudoGetXdot(ts, X, &Xdot));
  PetscCall(TSComputeIJacobian(ts, ts->ptime + ts->time_step, X, Xdot, 1. / ts->time_step, AA, BB, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(ts->vec_sol, &pseudo->update));
  PetscCall(VecDuplicate(ts->vec_sol, &pseudo->func));
  PetscCall(VecDuplicate(ts->vec_sol, &pseudo->xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSPseudoMonitorDefault(TS ts, PetscInt step, PetscReal ptime, Vec v, void *dummy)
{
  TS_Pseudo  *pseudo = (TS_Pseudo *)ts->data;
  PetscViewer viewer = (PetscViewer)dummy;

  PetscFunctionBegin;
  if (pseudo->fnorm < 0) { /* The last computed norm is stale, recompute */
    PetscCall(VecZeroEntries(pseudo->xdot));
    PetscCall(TSComputeIFunction(ts, ts->ptime, ts->vec_sol, pseudo->xdot, pseudo->func, PETSC_FALSE));
    PetscCall(PetscObjectStateGet((PetscObject)ts->vec_sol, &pseudo->Xstate));
    PetscCall(VecNorm(pseudo->func, NORM_2, &pseudo->fnorm));
  }
  PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject)ts)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "TS %" PetscInt_FMT " dt %g time %g fnorm %g\n", step, (double)ts->time_step, (double)ptime, (double)pseudo->fnorm));
  PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject)ts)->tablevel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_Pseudo(TS ts, PetscOptionItems PetscOptionsObject)
{
  TS_Pseudo  *pseudo = (TS_Pseudo *)ts->data;
  PetscBool   flg    = PETSC_FALSE;
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Pseudo-timestepping options");
  PetscCall(PetscOptionsBool("-ts_monitor_pseudo", "Monitor convergence", "", flg, &flg, NULL));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)ts), "stdout", &viewer));
    PetscCall(TSMonitorSet(ts, TSPseudoMonitorDefault, viewer, (PetscCtxDestroyFn *)PetscViewerDestroy));
  }
  flg = pseudo->increment_dt_from_initial_dt;
  PetscCall(PetscOptionsBool("-ts_pseudo_increment_dt_from_initial_dt", "Increase dt as a ratio from original dt", "TSPseudoIncrementDtFromInitialDt", flg, &flg, NULL));
  pseudo->increment_dt_from_initial_dt = flg;
  PetscCall(PetscOptionsReal("-ts_pseudo_increment", "Ratio to increase dt", "TSPseudoSetTimeStepIncrement", pseudo->dt_increment, &pseudo->dt_increment, NULL));
  PetscCall(PetscOptionsReal("-ts_pseudo_max_dt", "Maximum value for dt", "TSPseudoSetMaxTimeStep", pseudo->dt_max, &pseudo->dt_max, NULL));
  PetscCall(PetscOptionsReal("-ts_pseudo_fatol", "Tolerance for norm of function", "", pseudo->fatol, &pseudo->fatol, NULL));
  PetscCall(PetscOptionsReal("-ts_pseudo_frtol", "Relative tolerance for norm of function", "", pseudo->frtol, &pseudo->frtol, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_Pseudo(TS ts, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;
    PetscCall(PetscViewerASCIIPrintf(viewer, "  frtol - relative tolerance in function value %g\n", (double)pseudo->frtol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  fatol - absolute tolerance in function value %g\n", (double)pseudo->fatol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  dt_initial - initial timestep %g\n", (double)pseudo->dt_initial));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  dt_increment - increase in timestep on successful step %g\n", (double)pseudo->dt_increment));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  dt_max - maximum time %g\n", (double)pseudo->dt_max));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSPseudoSetVerifyTimeStep - Sets a user-defined routine to verify the quality of the
  last timestep.

  Logically Collective

  Input Parameters:
+ ts  - timestep context
. dt  - user-defined function to verify timestep
- ctx - [optional] user-defined context for private data for the timestep verification routine (may be `NULL`)

  Calling sequence of `func`:
+ ts     - the time-step context
. update - latest solution vector
. ctx    - [optional] user-defined timestep context
. newdt  - the timestep to use for the next step
- flag   - flag indicating whether the last time step was acceptable

  Level: advanced

  Note:
  The routine set here will be called by `TSPseudoVerifyTimeStep()`
  during the timestepping process.

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoVerifyTimeStepDefault()`, `TSPseudoVerifyTimeStep()`
@*/
PetscErrorCode TSPseudoSetVerifyTimeStep(TS ts, PetscErrorCode (*dt)(TS ts, Vec update, void *ctx, PetscReal *newdt, PetscBool *flag), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryMethod(ts, "TSPseudoSetVerifyTimeStep_C", (TS, PetscErrorCode (*)(TS, Vec, void *, PetscReal *, PetscBool *), void *), (ts, dt, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSPseudoSetTimeStepIncrement - Sets the scaling increment applied to
  dt when using the TSPseudoTimeStepDefault() routine.

  Logically Collective

  Input Parameters:
+ ts  - the timestep context
- inc - the scaling factor >= 1.0

  Options Database Key:
. -ts_pseudo_increment <increment> - set pseudo increment

  Level: advanced

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoSetTimeStep()`, `TSPseudoTimeStepDefault()`
@*/
PetscErrorCode TSPseudoSetTimeStepIncrement(TS ts, PetscReal inc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ts, inc, 2);
  PetscTryMethod(ts, "TSPseudoSetTimeStepIncrement_C", (TS, PetscReal), (ts, inc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSPseudoSetMaxTimeStep - Sets the maximum time step
  when using the TSPseudoTimeStepDefault() routine.

  Logically Collective

  Input Parameters:
+ ts    - the timestep context
- maxdt - the maximum time step, use a non-positive value to deactivate

  Options Database Key:
. -ts_pseudo_max_dt <increment> - set pseudo max dt

  Level: advanced

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoSetTimeStep()`, `TSPseudoTimeStepDefault()`
@*/
PetscErrorCode TSPseudoSetMaxTimeStep(TS ts, PetscReal maxdt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(ts, maxdt, 2);
  PetscTryMethod(ts, "TSPseudoSetMaxTimeStep_C", (TS, PetscReal), (ts, maxdt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSPseudoIncrementDtFromInitialDt - Indicates that a new timestep
  is computed via the formula $  dt = initial\_dt*initial\_fnorm/current\_fnorm $  rather than the default update, $  dt = current\_dt*previous\_fnorm/current\_fnorm.$

  Logically Collective

  Input Parameter:
. ts - the timestep context

  Options Database Key:
. -ts_pseudo_increment_dt_from_initial_dt <true,false> - use the initial $dt$ to determine increment

  Level: advanced

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoSetTimeStep()`, `TSPseudoTimeStepDefault()`
@*/
PetscErrorCode TSPseudoIncrementDtFromInitialDt(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryMethod(ts, "TSPseudoIncrementDtFromInitialDt_C", (TS), (ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSPseudoSetTimeStep - Sets the user-defined routine to be
  called at each pseudo-timestep to update the timestep.

  Logically Collective

  Input Parameters:
+ ts  - timestep context
. dt  - function to compute timestep
- ctx - [optional] user-defined context for private data required by the function (may be `NULL`)

  Calling sequence of `dt`:
+ ts    - the `TS` context
. newdt - the newly computed timestep
- ctx   - [optional] user-defined context

  Level: intermediate

  Notes:
  The routine set here will be called by `TSPseudoComputeTimeStep()`
  during the timestepping process.

  If not set then `TSPseudoTimeStepDefault()` is automatically used

.seealso: [](ch_ts), `TSPSEUDO`, `TSPseudoTimeStepDefault()`, `TSPseudoComputeTimeStep()`
@*/
PetscErrorCode TSPseudoSetTimeStep(TS ts, PetscErrorCode (*dt)(TS ts, PetscReal *newdt, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryMethod(ts, "TSPseudoSetTimeStep_C", (TS, PetscErrorCode (*)(TS, PetscReal *, void *), void *), (ts, dt, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ----------------------------------------------------------------------------- */

typedef PetscErrorCode (*FCN1)(TS, Vec, void *, PetscReal *, PetscBool *); /* force argument to next function to not be extern C*/
static PetscErrorCode TSPseudoSetVerifyTimeStep_Pseudo(TS ts, FCN1 dt, void *ctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  pseudo->verify    = dt;
  pseudo->verifyctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSPseudoSetTimeStepIncrement_Pseudo(TS ts, PetscReal inc)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  pseudo->dt_increment = inc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSPseudoSetMaxTimeStep_Pseudo(TS ts, PetscReal maxdt)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  pseudo->dt_max = maxdt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSPseudoIncrementDtFromInitialDt_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  pseudo->increment_dt_from_initial_dt = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef PetscErrorCode (*FCN2)(TS, PetscReal *, void *); /* force argument to next function to not be extern C*/
static PetscErrorCode TSPseudoSetTimeStep_Pseudo(TS ts, FCN2 dt, void *ctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;

  PetscFunctionBegin;
  pseudo->dt    = dt;
  pseudo->dtctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSPSEUDO - Solve steady state ODE and DAE problems with pseudo time stepping {cite}`ckk02` {cite}`kk97`

  This method solves equations of the form

  $$
  F(X,Xdot) = 0
  $$

  for steady state using the iteration

  $$
  [G'] S = -F(X,0)
  X += S
  $$

  where

  $$
  G(Y) = F(Y,(Y-X)/dt)
  $$

  This is linearly-implicit Euler with the residual always evaluated "at steady
  state".  See note below.

  In addition to the modified solve, a dedicated adaptive timestepping scheme is implemented, mimicking the switched evolution relaxation in {cite}`ckk02`.
  It determines the next timestep via

  $$
  dt_{n} = r dt_{n-1} \frac{\Vert F(X_{n},0) \Vert}{\Vert F(X_{n-1},0) \Vert}
  $$

  where $r$ is an additional growth factor (set by `-ts_pseudo_increment`).
  An alternative formulation is also available that uses the initial timestep and function norm.

  $$
  dt_{n} = r dt_{0} \frac{\Vert F(X_{n},0) \Vert}{\Vert F(X_{0},0) \Vert}
  $$

  This is chosen by setting `-ts_pseudo_increment_dt_from_initial_dt`.
  For either method, an upper limit on the timestep can be set by `-ts_pseudo_max_dt`.

  Options Database Keys:
+  -ts_pseudo_increment <real>                     - ratio of increase dt
.  -ts_pseudo_increment_dt_from_initial_dt <truth> - Increase dt as a ratio from original dt
.  -ts_pseudo_max_dt                               - Maximum dt for adaptive timestepping algorithm
.  -ts_pseudo_monitor                              - Monitor convergence of the function norm
.  -ts_pseudo_fatol <atol>                         - stop iterating when the function norm is less than `atol`
-  -ts_pseudo_frtol <rtol>                         - stop iterating when the function norm divided by the initial function norm is less than `rtol`

  Level: beginner

  Notes:
  The residual computed by this method includes the transient term (Xdot is computed instead of
  always being zero), but since the prediction from the last step is always the solution from the
  last step, on the first Newton iteration we have

  $$
  Xdot = (Xpredicted - Xold)/dt = (Xold-Xold)/dt = 0
  $$

  The Jacobian $ dF/dX + shift*dF/dXdot $ contains a non-zero $ dF/dXdot$ term. In the $ X' = F(X) $ case it
  is $ dF/dX + shift*1/dt $. Hence  still contains the $ 1/dt $ term so the Jacobian is not simply the
  Jacobian of $ F $ and thus this pseudo-transient continuation is not just Newton on $F(x)=0$.

  Therefore, the linear system solved by the first Newton iteration is equivalent to the one
  described above and in the papers.  If the user chooses to perform multiple Newton iterations, the
  algorithm is no longer the one described in the referenced papers.
  By default, the `SNESType` is set to `SNESKSPONLY` to match the algorithm from the referenced papers.
  Setting the `SNESType` via `-snes_type` will override this default setting.

.seealso: [](ch_ts), `TSCreate()`, `TS`, `TSSetType()`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_Pseudo(TS ts)
{
  TS_Pseudo *pseudo;
  SNES       snes;
  SNESType   stype;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Pseudo;
  ts->ops->destroy        = TSDestroy_Pseudo;
  ts->ops->view           = TSView_Pseudo;
  ts->ops->setup          = TSSetUp_Pseudo;
  ts->ops->step           = TSStep_Pseudo;
  ts->ops->setfromoptions = TSSetFromOptions_Pseudo;
  ts->ops->snesfunction   = SNESTSFormFunction_Pseudo;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Pseudo;
  ts->default_adapt_type  = TSADAPTNONE;
  ts->usessnes            = PETSC_TRUE;

  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetType(snes, &stype));
  if (!stype) PetscCall(SNESSetType(snes, SNESKSPONLY));

  PetscCall(PetscNew(&pseudo));
  ts->data = (void *)pseudo;

  pseudo->dt                           = TSPseudoTimeStepDefault;
  pseudo->dtctx                        = NULL;
  pseudo->dt_increment                 = 1.1;
  pseudo->increment_dt_from_initial_dt = PETSC_FALSE;
  pseudo->fnorm                        = -1;
  pseudo->fnorm_initial                = -1;
  pseudo->fnorm_previous               = -1;
#if defined(PETSC_USE_REAL_SINGLE)
  pseudo->fatol = 1.e-25;
  pseudo->frtol = 1.e-5;
#else
  pseudo->fatol = 1.e-50;
  pseudo->frtol = 1.e-12;
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetVerifyTimeStep_C", TSPseudoSetVerifyTimeStep_Pseudo));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetTimeStepIncrement_C", TSPseudoSetTimeStepIncrement_Pseudo));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetMaxTimeStep_C", TSPseudoSetMaxTimeStep_Pseudo));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoIncrementDtFromInitialDt_C", TSPseudoIncrementDtFromInitialDt_Pseudo));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSPseudoSetTimeStep_C", TSPseudoSetTimeStep_Pseudo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSPseudoTimeStepDefault - Default code to compute pseudo-timestepping.  Use with `TSPseudoSetTimeStep()`.

  Collective, No Fortran Support

  Input Parameters:
+ ts    - the timestep context
- dtctx - unused timestep context

  Output Parameter:
. newdt - the timestep to use for the next step

  Level: advanced

.seealso: [](ch_ts), `TSPseudoSetTimeStep()`, `TSPseudoComputeTimeStep()`, `TSPSEUDO`
@*/
PetscErrorCode TSPseudoTimeStepDefault(TS ts, PetscReal *newdt, void *dtctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo *)ts->data;
  PetscReal  inc    = pseudo->dt_increment;

  PetscFunctionBegin;
  if (pseudo->fnorm < 0.0) {
    PetscCall(VecZeroEntries(pseudo->xdot));
    PetscCall(TSComputeIFunction(ts, ts->ptime, ts->vec_sol, pseudo->xdot, pseudo->func, PETSC_FALSE));
    PetscCall(PetscObjectStateGet((PetscObject)ts->vec_sol, &pseudo->Xstate));
    PetscCall(VecNorm(pseudo->func, NORM_2, &pseudo->fnorm));
  }
  if (pseudo->fnorm_initial < 0) {
    /* first time through so compute initial function norm */
    pseudo->fnorm_initial  = pseudo->fnorm;
    pseudo->fnorm_previous = pseudo->fnorm;
  }
  if (pseudo->fnorm == 0.0) *newdt = 1.e12 * inc * ts->time_step;
  else if (pseudo->increment_dt_from_initial_dt) *newdt = inc * pseudo->dt_initial * pseudo->fnorm_initial / pseudo->fnorm;
  else *newdt = inc * ts->time_step * pseudo->fnorm_previous / pseudo->fnorm;
  if (pseudo->dt_max > 0) *newdt = PetscMin(*newdt, pseudo->dt_max);
  pseudo->fnorm_previous = pseudo->fnorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
