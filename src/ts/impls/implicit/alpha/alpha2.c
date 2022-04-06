/*
  Code for timestepping with implicit generalized-\alpha method
  for second order systems.
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@article{Chung1993,\n"
  "  title   = {A Time Integration Algorithm for Structural Dynamics with Improved Numerical Dissipation: The Generalized-$\\alpha$ Method},\n"
  "  author  = {J. Chung, G. M. Hubert},\n"
  "  journal = {ASME Journal of Applied Mechanics},\n"
  "  volume  = {60},\n"
  "  number  = {2},\n"
  "  pages   = {371--375},\n"
  "  year    = {1993},\n"
  "  issn    = {0021-8936},\n"
  "  doi     = {http://dx.doi.org/10.1115/1.2900803}\n}\n";

typedef struct {
  PetscReal stage_time;
  PetscReal shift_V;
  PetscReal shift_A;
  PetscReal scale_F;
  Vec       X0,Xa,X1;
  Vec       V0,Va,V1;
  Vec       A0,Aa,A1;

  Vec       vec_dot;

  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Gamma;
  PetscReal Beta;
  PetscInt  order;

  Vec       vec_sol_prev;
  Vec       vec_dot_prev;
  Vec       vec_lte_work[2];

  TSStepStatus status;
} TS_Alpha;

static PetscErrorCode TSAlpha_StageTime(TS ts)
{
  TS_Alpha  *th = (TS_Alpha*)ts->data;
  PetscReal t  = ts->ptime;
  PetscReal dt = ts->time_step;
  PetscReal Alpha_m = th->Alpha_m;
  PetscReal Alpha_f = th->Alpha_f;
  PetscReal Gamma   = th->Gamma;
  PetscReal Beta    = th->Beta;

  PetscFunctionBegin;
  th->stage_time = t + Alpha_f*dt;
  th->shift_V = Gamma/(dt*Beta);
  th->shift_A = Alpha_m/(Alpha_f*dt*dt*Beta);
  th->scale_F = 1/Alpha_f;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha_StageVecs(TS ts,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  Vec            X1 = X,      V1 = th->V1, A1 = th->A1;
  Vec            Xa = th->Xa, Va = th->Va, Aa = th->Aa;
  Vec            X0 = th->X0, V0 = th->V0, A0 = th->A0;
  PetscReal      dt = ts->time_step;
  PetscReal      Alpha_m = th->Alpha_m;
  PetscReal      Alpha_f = th->Alpha_f;
  PetscReal      Gamma   = th->Gamma;
  PetscReal      Beta    = th->Beta;

  PetscFunctionBegin;
  /* A1 = ... */
  PetscCall(VecWAXPY(A1,-1.0,X0,X1));
  PetscCall(VecAXPY (A1,-dt,V0));
  PetscCall(VecAXPBY(A1,-(1-2*Beta)/(2*Beta),1/(dt*dt*Beta),A0));
  /* V1 = ... */
  PetscCall(VecWAXPY(V1,(1.0-Gamma)/Gamma,A0,A1));
  PetscCall(VecAYPX (V1,dt*Gamma,V0));
  /* Xa = X0 + Alpha_f*(X1-X0) */
  PetscCall(VecWAXPY(Xa,-1.0,X0,X1));
  PetscCall(VecAYPX (Xa,Alpha_f,X0));
  /* Va = V0 + Alpha_f*(V1-V0) */
  PetscCall(VecWAXPY(Va,-1.0,V0,V1));
  PetscCall(VecAYPX (Va,Alpha_f,V0));
  /* Aa = A0 + Alpha_m*(A1-A0) */
  PetscCall(VecWAXPY(Aa,-1.0,A0,A1));
  PetscCall(VecAYPX (Aa,Alpha_m,A0));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha_SNESSolve(TS ts,Vec b,Vec x)
{
  PetscInt       nits,lits;

  PetscFunctionBegin;
  PetscCall(SNESSolve(ts->snes,b,x));
  PetscCall(SNESGetIterationNumber(ts->snes,&nits));
  PetscCall(SNESGetLinearSolveIterations(ts->snes,&lits));
  ts->snes_its += nits; ts->ksp_its += lits;
  PetscFunctionReturn(0);
}

/*
  Compute a consistent initial state for the generalized-alpha method.
  - Solve two successive backward Euler steps with halved time step.
  - Compute the initial second time derivative using backward differences.
  - If using adaptivity, estimate the LTE of the initial step.
*/
static PetscErrorCode TSAlpha_Restart(TS ts,PetscBool *initok)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      time_step;
  PetscReal      alpha_m,alpha_f,gamma,beta;
  Vec            X0 = ts->vec_sol, X1, X2 = th->X1;
  Vec            V0 = ts->vec_dot, V1, V2 = th->V1;
  PetscBool      stageok;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(X0,&X1));
  PetscCall(VecDuplicate(V0,&V1));

  /* Setup backward Euler with halved time step */
  PetscCall(TSAlpha2GetParams(ts,&alpha_m,&alpha_f,&gamma,&beta));
  PetscCall(TSAlpha2SetParams(ts,1,1,1,0.5));
  PetscCall(TSGetTimeStep(ts,&time_step));
  ts->time_step = time_step/2;
  PetscCall(TSAlpha_StageTime(ts));
  th->stage_time = ts->ptime;
  PetscCall(VecZeroEntries(th->A0));

  /* First BE step, (t0,X0,V0) -> (t1,X1,V1) */
  th->stage_time += ts->time_step;
  PetscCall(VecCopy(X0,th->X0));
  PetscCall(VecCopy(V0,th->V0));
  PetscCall(TSPreStage(ts,th->stage_time));
  PetscCall(VecCopy(th->X0,X1));
  PetscCall(TSAlpha_SNESSolve(ts,NULL,X1));
  PetscCall(VecCopy(th->V1,V1));
  PetscCall(TSPostStage(ts,th->stage_time,0,&X1));
  PetscCall(TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X1,&stageok));
  if (!stageok) goto finally;

  /* Second BE step, (t1,X1,V1) -> (t2,X2,V2) */
  th->stage_time += ts->time_step;
  PetscCall(VecCopy(X1,th->X0));
  PetscCall(VecCopy(V1,th->V0));
  PetscCall(TSPreStage(ts,th->stage_time));
  PetscCall(VecCopy(th->X0,X2));
  PetscCall(TSAlpha_SNESSolve(ts,NULL,X2));
  PetscCall(VecCopy(th->V1,V2));
  PetscCall(TSPostStage(ts,th->stage_time,0,&X2));
  PetscCall(TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X1,&stageok));
  if (!stageok) goto finally;

  /* Compute A0 ~ dV/dt at t0 with backward differences */
  PetscCall(VecZeroEntries(th->A0));
  PetscCall(VecAXPY(th->A0,-3/ts->time_step,V0));
  PetscCall(VecAXPY(th->A0,+4/ts->time_step,V1));
  PetscCall(VecAXPY(th->A0,-1/ts->time_step,V2));

  /* Rough, lower-order estimate LTE of the initial step */
  if (th->vec_lte_work[0]) {
    PetscCall(VecZeroEntries(th->vec_lte_work[0]));
    PetscCall(VecAXPY(th->vec_lte_work[0],+2,X2));
    PetscCall(VecAXPY(th->vec_lte_work[0],-4,X1));
    PetscCall(VecAXPY(th->vec_lte_work[0],+2,X0));
  }
  if (th->vec_lte_work[1]) {
    PetscCall(VecZeroEntries(th->vec_lte_work[1]));
    PetscCall(VecAXPY(th->vec_lte_work[1],+2,V2));
    PetscCall(VecAXPY(th->vec_lte_work[1],-4,V1));
    PetscCall(VecAXPY(th->vec_lte_work[1],+2,V0));
  }

 finally:
  /* Revert TSAlpha to the initial state (t0,X0,V0) */
  if (initok) *initok = stageok;
  PetscCall(TSSetTimeStep(ts,time_step));
  PetscCall(TSAlpha2SetParams(ts,alpha_m,alpha_f,gamma,beta));
  PetscCall(VecCopy(ts->vec_sol,th->X0));
  PetscCall(VecCopy(ts->vec_dot,th->V0));

  PetscCall(VecDestroy(&X1));
  PetscCall(VecDestroy(&V1));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscInt       rejections = 0;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation,&cited));

  if (!ts->steprollback) {
    if (th->vec_sol_prev) PetscCall(VecCopy(th->X0,th->vec_sol_prev));
    if (th->vec_dot_prev) PetscCall(VecCopy(th->V0,th->vec_dot_prev));
    PetscCall(VecCopy(ts->vec_sol,th->X0));
    PetscCall(VecCopy(ts->vec_dot,th->V0));
    PetscCall(VecCopy(th->A1,th->A0));
  }

  th->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && th->status != TS_STEP_COMPLETE) {

    if (ts->steprestart) {
      PetscCall(TSAlpha_Restart(ts,&stageok));
      if (!stageok) goto reject_step;
    }

    PetscCall(TSAlpha_StageTime(ts));
    PetscCall(VecCopy(th->X0,th->X1));
    PetscCall(TSPreStage(ts,th->stage_time));
    PetscCall(TSAlpha_SNESSolve(ts,NULL,th->X1));
    PetscCall(TSPostStage(ts,th->stage_time,0,&th->Xa));
    PetscCall(TSAdaptCheckStage(ts->adapt,ts,th->stage_time,th->Xa,&stageok));
    if (!stageok) goto reject_step;

    th->status = TS_STEP_PENDING;
    PetscCall(VecCopy(th->X1,ts->vec_sol));
    PetscCall(VecCopy(th->V1,ts->vec_dot));
    PetscCall(TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept));
    th->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      PetscCall(VecCopy(th->X0,ts->vec_sol));
      PetscCall(VecCopy(th->V0,ts->vec_dot));
      ts->time_step = next_time_step;
      goto reject_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscCall(PetscInfo(ts,"Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n",ts->steps,rejections));
    }

  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateWLTE_Alpha(TS ts,NormType wnormtype,PetscInt *order,PetscReal *wlte)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  Vec            X = th->X1;              /* X = solution */
  Vec            V = th->V1;              /* V = solution */
  Vec            Y = th->vec_lte_work[0]; /* Y = X + LTE  */
  Vec            Z = th->vec_lte_work[1]; /* Z = V + LTE  */
  PetscReal      enormX,enormV,enormXa,enormVa,enormXr,enormVr;

  PetscFunctionBegin;
  if (!th->vec_sol_prev) {*wlte = -1; PetscFunctionReturn(0);}
  if (!th->vec_dot_prev) {*wlte = -1; PetscFunctionReturn(0);}
  if (!th->vec_lte_work[0]) {*wlte = -1; PetscFunctionReturn(0);}
  if (!th->vec_lte_work[1]) {*wlte = -1; PetscFunctionReturn(0);}
  if (ts->steprestart) {
    /* th->vec_lte_prev is set to the LTE in TSAlpha_Restart() */
    PetscCall(VecAXPY(Y,1,X));
    PetscCall(VecAXPY(Z,1,V));
  } else {
    /* Compute LTE using backward differences with non-constant time step */
    PetscReal   h = ts->time_step, h_prev = ts->ptime - ts->ptime_prev;
    PetscReal   a = 1 + h_prev/h;
    PetscScalar scal[3]; Vec vecX[3],vecV[3];
    scal[0] = +1/a;   scal[1] = -1/(a-1); scal[2] = +1/(a*(a-1));
    vecX[0] = th->X1; vecX[1] = th->X0;   vecX[2] = th->vec_sol_prev;
    vecV[0] = th->V1; vecV[1] = th->V0;   vecV[2] = th->vec_dot_prev;
    PetscCall(VecCopy(X,Y));
    PetscCall(VecMAXPY(Y,3,scal,vecX));
    PetscCall(VecCopy(V,Z));
    PetscCall(VecMAXPY(Z,3,scal,vecV));
  }
  /* XXX ts->atol and ts->vatol are not appropriate for computing enormV */
  PetscCall(TSErrorWeightedNorm(ts,X,Y,wnormtype,&enormX,&enormXa,&enormXr));
  PetscCall(TSErrorWeightedNorm(ts,V,Z,wnormtype,&enormV,&enormVa,&enormVr));
  if (wnormtype == NORM_2)
    *wlte = PetscSqrtReal(PetscSqr(enormX)/2 + PetscSqr(enormV)/2);
  else
    *wlte = PetscMax(enormX,enormV);
  if (order) *order = 2;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  PetscCall(VecCopy(th->X0,ts->vec_sol));
  PetscCall(VecCopy(th->V0,ts->vec_dot));
  PetscFunctionReturn(0);
}

/*
static PetscErrorCode TSInterpolate_Alpha(TS ts,PetscReal t,Vec X,Vec V)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      dt  = t - ts->ptime;

  PetscFunctionBegin;
  PetscCall(VecCopy(ts->vec_dot,V));
  PetscCall(VecAXPY(V,dt*(1-th->Gamma),th->A0));
  PetscCall(VecAXPY(V,dt*th->Gamma,th->A1));
  PetscCall(VecCopy(ts->vec_sol,X));
  PetscCall(VecAXPY(X,dt,V));
  PetscCall(VecAXPY(X,dt*dt*((PetscReal)0.5-th->Beta),th->A0));
  PetscCall(VecAXPY(X,dt*dt*th->Beta,th->A1));
  PetscFunctionReturn(0);
}
*/

static PetscErrorCode SNESTSFormFunction_Alpha(PETSC_UNUSED SNES snes,Vec X,Vec F,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va, Aa = th->Aa;

  PetscFunctionBegin;
  PetscCall(TSAlpha_StageVecs(ts,X));
  /* F = Function(ta,Xa,Va,Aa) */
  PetscCall(TSComputeI2Function(ts,ta,Xa,Va,Aa,F));
  PetscCall(VecScale(F,th->scale_F));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_Alpha(PETSC_UNUSED SNES snes,PETSC_UNUSED Vec X,Mat J,Mat P,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va, Aa = th->Aa;
  PetscReal      dVdX = th->shift_V, dAdX = th->shift_A;

  PetscFunctionBegin;
  /* J,P = Jacobian(ta,Xa,Va,Aa) */
  PetscCall(TSComputeI2Jacobian(ts,ta,Xa,Va,Aa,dVdX,dAdX,J,P));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&th->X0));
  PetscCall(VecDestroy(&th->Xa));
  PetscCall(VecDestroy(&th->X1));
  PetscCall(VecDestroy(&th->V0));
  PetscCall(VecDestroy(&th->Va));
  PetscCall(VecDestroy(&th->V1));
  PetscCall(VecDestroy(&th->A0));
  PetscCall(VecDestroy(&th->Aa));
  PetscCall(VecDestroy(&th->A1));
  PetscCall(VecDestroy(&th->vec_sol_prev));
  PetscCall(VecDestroy(&th->vec_dot_prev));
  PetscCall(VecDestroy(&th->vec_lte_work[0]));
  PetscCall(VecDestroy(&th->vec_lte_work[1]));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Alpha(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_Alpha(ts));
  PetscCall(PetscFree(ts->data));

  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetRadius_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetParams_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2GetParams_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      match;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(ts->vec_sol,&th->X0));
  PetscCall(VecDuplicate(ts->vec_sol,&th->Xa));
  PetscCall(VecDuplicate(ts->vec_sol,&th->X1));
  PetscCall(VecDuplicate(ts->vec_sol,&th->V0));
  PetscCall(VecDuplicate(ts->vec_sol,&th->Va));
  PetscCall(VecDuplicate(ts->vec_sol,&th->V1));
  PetscCall(VecDuplicate(ts->vec_sol,&th->A0));
  PetscCall(VecDuplicate(ts->vec_sol,&th->Aa));
  PetscCall(VecDuplicate(ts->vec_sol,&th->A1));

  PetscCall(TSGetAdapt(ts,&ts->adapt));
  PetscCall(TSAdaptCandidatesClear(ts->adapt));
  PetscCall(PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&match));
  if (!match) {
    PetscCall(VecDuplicate(ts->vec_sol,&th->vec_sol_prev));
    PetscCall(VecDuplicate(ts->vec_sol,&th->vec_dot_prev));
    PetscCall(VecDuplicate(ts->vec_sol,&th->vec_lte_work[0]));
    PetscCall(VecDuplicate(ts->vec_sol,&th->vec_lte_work[1]));
  }

  PetscCall(TSGetSNES(ts,&ts->snes));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_Alpha(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Generalized-Alpha ODE solver options");
  {
    PetscBool flg;
    PetscReal radius = 1;
    PetscCall(PetscOptionsReal("-ts_alpha_radius","Spectral radius (high-frequency dissipation)","TSAlpha2SetRadius",radius,&radius,&flg));
    if (flg) PetscCall(TSAlpha2SetRadius(ts,radius));
    PetscCall(PetscOptionsReal("-ts_alpha_alpha_m","Algorithmic parameter alpha_m","TSAlpha2SetParams",th->Alpha_m,&th->Alpha_m,NULL));
    PetscCall(PetscOptionsReal("-ts_alpha_alpha_f","Algorithmic parameter alpha_f","TSAlpha2SetParams",th->Alpha_f,&th->Alpha_f,NULL));
    PetscCall(PetscOptionsReal("-ts_alpha_gamma","Algorithmic parameter gamma","TSAlpha2SetParams",th->Gamma,&th->Gamma,NULL));
    PetscCall(PetscOptionsReal("-ts_alpha_beta","Algorithmic parameter beta","TSAlpha2SetParams",th->Beta,&th->Beta,NULL));
    PetscCall(TSAlpha2SetParams(ts,th->Alpha_m,th->Alpha_f,th->Gamma,th->Beta));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Alpha(TS ts,PetscViewer viewer)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Alpha_m=%g, Alpha_f=%g, Gamma=%g, Beta=%g\n",(double)th->Alpha_m,(double)th->Alpha_f,(double)th->Gamma,(double)th->Beta));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha2SetRadius_Alpha(TS ts,PetscReal radius)
{
  PetscReal      alpha_m,alpha_f,gamma,beta;

  PetscFunctionBegin;
  PetscCheckFalse(radius < 0 || radius > 1,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  alpha_m = (2-radius)/(1+radius);
  alpha_f = 1/(1+radius);
  gamma   = (PetscReal)0.5 + alpha_m - alpha_f;
  beta    = (PetscReal)0.5 * (1 + alpha_m - alpha_f); beta *= beta;
  PetscCall(TSAlpha2SetParams(ts,alpha_m,alpha_f,gamma,beta));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha2SetParams_Alpha(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma,PetscReal beta)
{
  TS_Alpha  *th = (TS_Alpha*)ts->data;
  PetscReal tol = 100*PETSC_MACHINE_EPSILON;
  PetscReal res = ((PetscReal)0.5 + alpha_m - alpha_f) - gamma;

  PetscFunctionBegin;
  th->Alpha_m = alpha_m;
  th->Alpha_f = alpha_f;
  th->Gamma   = gamma;
  th->Beta    = beta;
  th->order   = (PetscAbsReal(res) < tol) ? 2 : 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha2GetParams_Alpha(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma,PetscReal *beta)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (alpha_m) *alpha_m = th->Alpha_m;
  if (alpha_f) *alpha_f = th->Alpha_f;
  if (gamma)   *gamma   = th->Gamma;
  if (beta)    *beta    = th->Beta;
  PetscFunctionReturn(0);
}

/*MC
      TSALPHA2 - ODE/DAE solver using the implicit Generalized-Alpha method
                 for second-order systems

  Level: beginner

  References:
. * - J. Chung, G.M.Hubert. "A Time Integration Algorithm for Structural
  Dynamics with Improved Numerical Dissipation: The Generalized-alpha
  Method" ASME Journal of Applied Mechanics, 60, 371:375, 1993.

.seealso:  TS, TSCreate(), TSSetType(), TSAlpha2SetRadius(), TSAlpha2SetParams()
M*/
PETSC_EXTERN PetscErrorCode TSCreate_Alpha2(TS ts)
{
  TS_Alpha       *th;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Alpha;
  ts->ops->destroy        = TSDestroy_Alpha;
  ts->ops->view           = TSView_Alpha;
  ts->ops->setup          = TSSetUp_Alpha;
  ts->ops->setfromoptions = TSSetFromOptions_Alpha;
  ts->ops->step           = TSStep_Alpha;
  ts->ops->evaluatewlte   = TSEvaluateWLTE_Alpha;
  ts->ops->rollback       = TSRollBack_Alpha;
  /*ts->ops->interpolate  = TSInterpolate_Alpha;*/
  ts->ops->snesfunction   = SNESTSFormFunction_Alpha;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Alpha;
  ts->default_adapt_type  = TSADAPTNONE;

  ts->usessnes = PETSC_TRUE;

  PetscCall(PetscNewLog(ts,&th));
  ts->data = (void*)th;

  th->Alpha_m = 0.5;
  th->Alpha_f = 0.5;
  th->Gamma   = 0.5;
  th->Beta    = 0.25;
  th->order   = 2;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetRadius_C",TSAlpha2SetRadius_Alpha));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetParams_C",TSAlpha2SetParams_Alpha));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2GetParams_C",TSAlpha2GetParams_Alpha));
  PetscFunctionReturn(0);
}

/*@
  TSAlpha2SetRadius - sets the desired spectral radius of the method
                      (i.e. high-frequency numerical damping)

  Logically Collective on TS

  The algorithmic parameters \alpha_m and \alpha_f of the
  generalized-\alpha method can be computed in terms of a specified
  spectral radius \rho in [0,1] for infinite time step in order to
  control high-frequency numerical damping:
    \alpha_m = (2-\rho)/(1+\rho)
    \alpha_f = 1/(1+\rho)

  Input Parameters:
+  ts - timestepping context
-  radius - the desired spectral radius

  Options Database:
.  -ts_alpha_radius <radius> - set the desired spectral radius

  Level: intermediate

.seealso: TSAlpha2SetParams(), TSAlpha2GetParams()
@*/
PetscErrorCode TSAlpha2SetRadius(TS ts,PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,radius,2);
  PetscCheckFalse(radius < 0 || radius > 1,((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  PetscTryMethod(ts,"TSAlpha2SetRadius_C",(TS,PetscReal),(ts,radius));
  PetscFunctionReturn(0);
}

/*@
  TSAlpha2SetParams - sets the algorithmic parameters for TSALPHA2

  Logically Collective on TS

  Second-order accuracy can be obtained so long as:
    \gamma = 1/2 + alpha_m - alpha_f
    \beta  = 1/4 (1 + alpha_m - alpha_f)^2

  Unconditional stability requires:
    \alpha_m >= \alpha_f >= 1/2

  Input Parameters:
+ ts - timestepping context
. \alpha_m - algorithmic parameter
. \alpha_f - algorithmic parameter
. \gamma   - algorithmic parameter
- \beta    - algorithmic parameter

  Options Database:
+ -ts_alpha_alpha_m <alpha_m> - set alpha_m
. -ts_alpha_alpha_f <alpha_f> - set alpha_f
. -ts_alpha_gamma   <gamma> - set gamma
- -ts_alpha_beta    <beta> - set beta

  Note:
  Use of this function is normally only required to hack TSALPHA2 to
  use a modified integration scheme. Users should call
  TSAlpha2SetRadius() to set the desired spectral radius of the methods
  (i.e. high-frequency damping) in order so select optimal values for
  these parameters.

  Level: advanced

.seealso: TSAlpha2SetRadius(), TSAlpha2GetParams()
@*/
PetscErrorCode TSAlpha2SetParams(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma,PetscReal beta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,alpha_m,2);
  PetscValidLogicalCollectiveReal(ts,alpha_f,3);
  PetscValidLogicalCollectiveReal(ts,gamma,4);
  PetscValidLogicalCollectiveReal(ts,beta,5);
  PetscTryMethod(ts,"TSAlpha2SetParams_C",(TS,PetscReal,PetscReal,PetscReal,PetscReal),(ts,alpha_m,alpha_f,gamma,beta));
  PetscFunctionReturn(0);
}

/*@
  TSAlpha2GetParams - gets the algorithmic parameters for TSALPHA2

  Not Collective

  Input Parameter:
. ts - timestepping context

  Output Parameters:
+ \alpha_m - algorithmic parameter
. \alpha_f - algorithmic parameter
. \gamma   - algorithmic parameter
- \beta    - algorithmic parameter

  Note:
  Use of this function is normally only required to hack TSALPHA2 to
  use a modified integration scheme. Users should call
  TSAlpha2SetRadius() to set the high-frequency damping (i.e. spectral
  radius of the method) in order so select optimal values for these
  parameters.

  Level: advanced

.seealso: TSAlpha2SetRadius(), TSAlpha2SetParams()
@*/
PetscErrorCode TSAlpha2GetParams(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma,PetscReal *beta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (alpha_m) PetscValidRealPointer(alpha_m,2);
  if (alpha_f) PetscValidRealPointer(alpha_f,3);
  if (gamma)   PetscValidRealPointer(gamma,4);
  if (beta)    PetscValidRealPointer(beta,5);
  PetscUseMethod(ts,"TSAlpha2GetParams_C",(TS,PetscReal*,PetscReal*,PetscReal*,PetscReal*),(ts,alpha_m,alpha_f,gamma,beta));
  PetscFunctionReturn(0);
}
