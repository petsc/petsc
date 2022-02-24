/*
  Code for timestepping with implicit generalized-\alpha method
  for first order systems.
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@article{Jansen2000,\n"
  "  title   = {A generalized-$\\alpha$ method for integrating the filtered {N}avier--{S}tokes equations with a stabilized finite element method},\n"
  "  author  = {Kenneth E. Jansen and Christian H. Whiting and Gregory M. Hulbert},\n"
  "  journal = {Computer Methods in Applied Mechanics and Engineering},\n"
  "  volume  = {190},\n"
  "  number  = {3--4},\n"
  "  pages   = {305--319},\n"
  "  year    = {2000},\n"
  "  issn    = {0045-7825},\n"
  "  doi     = {http://dx.doi.org/10.1016/S0045-7825(00)00203-6}\n}\n";

typedef struct {
  PetscReal stage_time;
  PetscReal shift_V;
  PetscReal scale_F;
  Vec       X0,Xa,X1;
  Vec       V0,Va,V1;

  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Gamma;
  PetscInt  order;

  Vec       vec_sol_prev;
  Vec       vec_lte_work;

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

  PetscFunctionBegin;
  th->stage_time = t + Alpha_f*dt;
  th->shift_V = Alpha_m/(Alpha_f*Gamma*dt);
  th->scale_F = 1/Alpha_f;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha_StageVecs(TS ts,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  Vec            X1 = X,      V1 = th->V1;
  Vec            Xa = th->Xa, Va = th->Va;
  Vec            X0 = th->X0, V0 = th->V0;
  PetscReal      dt = ts->time_step;
  PetscReal      Alpha_m = th->Alpha_m;
  PetscReal      Alpha_f = th->Alpha_f;
  PetscReal      Gamma   = th->Gamma;

  PetscFunctionBegin;
  /* V1 = 1/(Gamma*dT)*(X1-X0) + (1-1/Gamma)*V0 */
  CHKERRQ(VecWAXPY(V1,-1.0,X0,X1));
  CHKERRQ(VecAXPBY(V1,1-1/Gamma,1/(Gamma*dt),V0));
  /* Xa = X0 + Alpha_f*(X1-X0) */
  CHKERRQ(VecWAXPY(Xa,-1.0,X0,X1));
  CHKERRQ(VecAYPX(Xa,Alpha_f,X0));
  /* Va = V0 + Alpha_m*(V1-V0) */
  CHKERRQ(VecWAXPY(Va,-1.0,V0,V1));
  CHKERRQ(VecAYPX(Va,Alpha_m,V0));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha_SNESSolve(TS ts,Vec b,Vec x)
{
  PetscInt       nits,lits;

  PetscFunctionBegin;
  CHKERRQ(SNESSolve(ts->snes,b,x));
  CHKERRQ(SNESGetIterationNumber(ts->snes,&nits));
  CHKERRQ(SNESGetLinearSolveIterations(ts->snes,&lits));
  ts->snes_its += nits; ts->ksp_its += lits;
  PetscFunctionReturn(0);
}

/*
  Compute a consistent initial state for the generalized-alpha method.
  - Solve two successive backward Euler steps with halved time step.
  - Compute the initial time derivative using backward differences.
  - If using adaptivity, estimate the LTE of the initial step.
*/
static PetscErrorCode TSAlpha_Restart(TS ts,PetscBool *initok)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      time_step;
  PetscReal      alpha_m,alpha_f,gamma;
  Vec            X0 = ts->vec_sol, X1, X2 = th->X1;
  PetscBool      stageok;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(X0,&X1));

  /* Setup backward Euler with halved time step */
  CHKERRQ(TSAlphaGetParams(ts,&alpha_m,&alpha_f,&gamma));
  CHKERRQ(TSAlphaSetParams(ts,1,1,1));
  CHKERRQ(TSGetTimeStep(ts,&time_step));
  ts->time_step = time_step/2;
  CHKERRQ(TSAlpha_StageTime(ts));
  th->stage_time = ts->ptime;
  CHKERRQ(VecZeroEntries(th->V0));

  /* First BE step, (t0,X0) -> (t1,X1) */
  th->stage_time += ts->time_step;
  CHKERRQ(VecCopy(X0,th->X0));
  CHKERRQ(TSPreStage(ts,th->stage_time));
  CHKERRQ(VecCopy(th->X0,X1));
  CHKERRQ(TSAlpha_SNESSolve(ts,NULL,X1));
  CHKERRQ(TSPostStage(ts,th->stage_time,0,&X1));
  CHKERRQ(TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X1,&stageok));
  if (!stageok) goto finally;

  /* Second BE step, (t1,X1) -> (t2,X2) */
  th->stage_time += ts->time_step;
  CHKERRQ(VecCopy(X1,th->X0));
  CHKERRQ(TSPreStage(ts,th->stage_time));
  CHKERRQ(VecCopy(th->X0,X2));
  CHKERRQ(TSAlpha_SNESSolve(ts,NULL,X2));
  CHKERRQ(TSPostStage(ts,th->stage_time,0,&X2));
  CHKERRQ(TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X2,&stageok));
  if (!stageok) goto finally;

  /* Compute V0 ~ dX/dt at t0 with backward differences */
  CHKERRQ(VecZeroEntries(th->V0));
  CHKERRQ(VecAXPY(th->V0,-3/ts->time_step,X0));
  CHKERRQ(VecAXPY(th->V0,+4/ts->time_step,X1));
  CHKERRQ(VecAXPY(th->V0,-1/ts->time_step,X2));

  /* Rough, lower-order estimate LTE of the initial step */
  if (th->vec_lte_work) {
    CHKERRQ(VecZeroEntries(th->vec_lte_work));
    CHKERRQ(VecAXPY(th->vec_lte_work,+2,X2));
    CHKERRQ(VecAXPY(th->vec_lte_work,-4,X1));
    CHKERRQ(VecAXPY(th->vec_lte_work,+2,X0));
  }

 finally:
  /* Revert TSAlpha to the initial state (t0,X0) */
  if (initok) *initok = stageok;
  CHKERRQ(TSSetTimeStep(ts,time_step));
  CHKERRQ(TSAlphaSetParams(ts,alpha_m,alpha_f,gamma));
  CHKERRQ(VecCopy(ts->vec_sol,th->X0));

  CHKERRQ(VecDestroy(&X1));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscInt       rejections = 0;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));

  if (!ts->steprollback) {
    if (th->vec_sol_prev) CHKERRQ(VecCopy(th->X0,th->vec_sol_prev));
    CHKERRQ(VecCopy(ts->vec_sol,th->X0));
    CHKERRQ(VecCopy(th->V1,th->V0));
  }

  th->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && th->status != TS_STEP_COMPLETE) {

    if (ts->steprestart) {
      CHKERRQ(TSAlpha_Restart(ts,&stageok));
      if (!stageok) goto reject_step;
    }

    CHKERRQ(TSAlpha_StageTime(ts));
    CHKERRQ(VecCopy(th->X0,th->X1));
    CHKERRQ(TSPreStage(ts,th->stage_time));
    CHKERRQ(TSAlpha_SNESSolve(ts,NULL,th->X1));
    CHKERRQ(TSPostStage(ts,th->stage_time,0,&th->Xa));
    CHKERRQ(TSAdaptCheckStage(ts->adapt,ts,th->stage_time,th->Xa,&stageok));
    if (!stageok) goto reject_step;

    th->status = TS_STEP_PENDING;
    CHKERRQ(VecCopy(th->X1,ts->vec_sol));
    CHKERRQ(TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept));
    th->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      CHKERRQ(VecCopy(th->X0,ts->vec_sol));
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
      CHKERRQ(PetscInfo(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections));
    }

  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateWLTE_Alpha(TS ts,NormType wnormtype,PetscInt *order,PetscReal *wlte)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  Vec            X = th->X1;           /* X = solution */
  Vec            Y = th->vec_lte_work; /* Y = X + LTE  */
  PetscReal      wltea,wlter;

  PetscFunctionBegin;
  if (!th->vec_sol_prev) {*wlte = -1; PetscFunctionReturn(0);}
  if (!th->vec_lte_work) {*wlte = -1; PetscFunctionReturn(0);}
  if (ts->steprestart) {
    /* th->vec_lte_work is set to the LTE in TSAlpha_Restart() */
    CHKERRQ(VecAXPY(Y,1,X));
  } else {
    /* Compute LTE using backward differences with non-constant time step */
    PetscReal   h = ts->time_step, h_prev = ts->ptime - ts->ptime_prev;
    PetscReal   a = 1 + h_prev/h;
    PetscScalar scal[3]; Vec vecs[3];
    scal[0] = +1/a;   scal[1] = -1/(a-1); scal[2] = +1/(a*(a-1));
    vecs[0] = th->X1; vecs[1] = th->X0;   vecs[2] = th->vec_sol_prev;
    CHKERRQ(VecCopy(X,Y));
    CHKERRQ(VecMAXPY(Y,3,scal,vecs));
  }
  CHKERRQ(TSErrorWeightedNorm(ts,X,Y,wnormtype,wlte,&wltea,&wlter));
  if (order) *order = 2;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(VecCopy(th->X0,ts->vec_sol));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_Alpha(TS ts,PetscReal t,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      dt  = t - ts->ptime;

  PetscFunctionBegin;
  CHKERRQ(VecCopy(ts->vec_sol,X));
  CHKERRQ(VecAXPY(X,th->Gamma*dt,th->V1));
  CHKERRQ(VecAXPY(X,(1-th->Gamma)*dt,th->V0));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormFunction_Alpha(PETSC_UNUSED SNES snes,Vec X,Vec F,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va;

  PetscFunctionBegin;
  CHKERRQ(TSAlpha_StageVecs(ts,X));
  /* F = Function(ta,Xa,Va) */
  CHKERRQ(TSComputeIFunction(ts,ta,Xa,Va,F,PETSC_FALSE));
  CHKERRQ(VecScale(F,th->scale_F));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_Alpha(PETSC_UNUSED SNES snes,PETSC_UNUSED Vec X,Mat J,Mat P,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va;
  PetscReal      dVdX = th->shift_V;

  PetscFunctionBegin;
  /* J,P = Jacobian(ta,Xa,Va) */
  CHKERRQ(TSComputeIJacobian(ts,ta,Xa,Va,dVdX,J,P,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&th->X0));
  CHKERRQ(VecDestroy(&th->Xa));
  CHKERRQ(VecDestroy(&th->X1));
  CHKERRQ(VecDestroy(&th->V0));
  CHKERRQ(VecDestroy(&th->Va));
  CHKERRQ(VecDestroy(&th->V1));
  CHKERRQ(VecDestroy(&th->vec_sol_prev));
  CHKERRQ(VecDestroy(&th->vec_lte_work));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Alpha(TS ts)
{
  PetscFunctionBegin;
  CHKERRQ(TSReset_Alpha(ts));
  CHKERRQ(PetscFree(ts->data));

  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetRadius_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetParams_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSAlphaGetParams_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      match;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(ts->vec_sol,&th->X0));
  CHKERRQ(VecDuplicate(ts->vec_sol,&th->Xa));
  CHKERRQ(VecDuplicate(ts->vec_sol,&th->X1));
  CHKERRQ(VecDuplicate(ts->vec_sol,&th->V0));
  CHKERRQ(VecDuplicate(ts->vec_sol,&th->Va));
  CHKERRQ(VecDuplicate(ts->vec_sol,&th->V1));

  CHKERRQ(TSGetAdapt(ts,&ts->adapt));
  CHKERRQ(TSAdaptCandidatesClear(ts->adapt));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&match));
  if (!match) {
    CHKERRQ(VecDuplicate(ts->vec_sol,&th->vec_sol_prev));
    CHKERRQ(VecDuplicate(ts->vec_sol,&th->vec_lte_work));
  }

  CHKERRQ(TSGetSNES(ts,&ts->snes));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_Alpha(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Generalized-Alpha ODE solver options"));
  {
    PetscBool flg;
    PetscReal radius = 1;
    CHKERRQ(PetscOptionsReal("-ts_alpha_radius","Spectral radius (high-frequency dissipation)","TSAlphaSetRadius",radius,&radius,&flg));
    if (flg) CHKERRQ(TSAlphaSetRadius(ts,radius));
    CHKERRQ(PetscOptionsReal("-ts_alpha_alpha_m","Algorithmic parameter alpha_m","TSAlphaSetParams",th->Alpha_m,&th->Alpha_m,NULL));
    CHKERRQ(PetscOptionsReal("-ts_alpha_alpha_f","Algorithmic parameter alpha_f","TSAlphaSetParams",th->Alpha_f,&th->Alpha_f,NULL));
    CHKERRQ(PetscOptionsReal("-ts_alpha_gamma","Algorithmic parameter gamma","TSAlphaSetParams",th->Gamma,&th->Gamma,NULL));
    CHKERRQ(TSAlphaSetParams(ts,th->Alpha_m,th->Alpha_f,th->Gamma));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Alpha(TS ts,PetscViewer viewer)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Alpha_m=%g, Alpha_f=%g, Gamma=%g\n",(double)th->Alpha_m,(double)th->Alpha_f,(double)th->Gamma));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlphaSetRadius_Alpha(TS ts,PetscReal radius)
{
  PetscReal      alpha_m,alpha_f,gamma;

  PetscFunctionBegin;
  PetscCheckFalse(radius < 0 || radius > 1,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  alpha_m = (PetscReal)0.5*(3-radius)/(1+radius);
  alpha_f = 1/(1+radius);
  gamma   = (PetscReal)0.5 + alpha_m - alpha_f;
  CHKERRQ(TSAlphaSetParams(ts,alpha_m,alpha_f,gamma));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlphaSetParams_Alpha(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma)
{
  TS_Alpha  *th = (TS_Alpha*)ts->data;
  PetscReal tol = 100*PETSC_MACHINE_EPSILON;
  PetscReal res = ((PetscReal)0.5 + alpha_m - alpha_f) - gamma;

  PetscFunctionBegin;
  th->Alpha_m = alpha_m;
  th->Alpha_f = alpha_f;
  th->Gamma   = gamma;
  th->order   = (PetscAbsReal(res) < tol) ? 2 : 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlphaGetParams_Alpha(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (alpha_m) *alpha_m = th->Alpha_m;
  if (alpha_f) *alpha_f = th->Alpha_f;
  if (gamma)   *gamma   = th->Gamma;
  PetscFunctionReturn(0);
}

/*MC
      TSALPHA - ODE/DAE solver using the implicit Generalized-Alpha method
                for first-order systems

  Level: beginner

  References:
+ * - K.E. Jansen, C.H. Whiting, G.M. Hulber, "A generalized-alpha
  method for integrating the filtered Navier-Stokes equations with a
  stabilized finite element method", Computer Methods in Applied
  Mechanics and Engineering, 190, 305-319, 2000.
  DOI: 10.1016/S0045-7825(00)00203-6.
- * -  J. Chung, G.M.Hubert. "A Time Integration Algorithm for Structural
  Dynamics with Improved Numerical Dissipation: The Generalized-alpha
  Method" ASME Journal of Applied Mechanics, 60, 371:375, 1993.

.seealso:  TS, TSCreate(), TSSetType(), TSAlphaSetRadius(), TSAlphaSetParams()
M*/
PETSC_EXTERN PetscErrorCode TSCreate_Alpha(TS ts)
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
  ts->ops->interpolate    = TSInterpolate_Alpha;
  ts->ops->snesfunction   = SNESTSFormFunction_Alpha;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Alpha;
  ts->default_adapt_type  = TSADAPTNONE;

  ts->usessnes = PETSC_TRUE;

  CHKERRQ(PetscNewLog(ts,&th));
  ts->data = (void*)th;

  th->Alpha_m = 0.5;
  th->Alpha_f = 0.5;
  th->Gamma   = 0.5;
  th->order   = 2;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetRadius_C",TSAlphaSetRadius_Alpha));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetParams_C",TSAlphaSetParams_Alpha));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSAlphaGetParams_C",TSAlphaGetParams_Alpha));
  PetscFunctionReturn(0);
}

/*@
  TSAlphaSetRadius - sets the desired spectral radius of the method
                     (i.e. high-frequency numerical damping)

  Logically Collective on TS

  The algorithmic parameters \alpha_m and \alpha_f of the
  generalized-\alpha method can be computed in terms of a specified
  spectral radius \rho in [0,1] for infinite time step in order to
  control high-frequency numerical damping:
    \alpha_m = 0.5*(3-\rho)/(1+\rho)
    \alpha_f = 1/(1+\rho)

  Input Parameters:
+  ts - timestepping context
-  radius - the desired spectral radius

  Options Database:
.  -ts_alpha_radius <radius> - set alpha radius

  Level: intermediate

.seealso: TSAlphaSetParams(), TSAlphaGetParams()
@*/
PetscErrorCode TSAlphaSetRadius(TS ts,PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,radius,2);
  PetscCheckFalse(radius < 0 || radius > 1,((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  CHKERRQ(PetscTryMethod(ts,"TSAlphaSetRadius_C",(TS,PetscReal),(ts,radius)));
  PetscFunctionReturn(0);
}

/*@
  TSAlphaSetParams - sets the algorithmic parameters for TSALPHA

  Logically Collective on TS

  Second-order accuracy can be obtained so long as:
    \gamma = 0.5 + alpha_m - alpha_f

  Unconditional stability requires:
    \alpha_m >= \alpha_f >= 0.5

  Backward Euler method is recovered with:
    \alpha_m = \alpha_f = gamma = 1

  Input Parameters:
+  ts - timestepping context
.  \alpha_m - algorithmic parameter
.  \alpha_f - algorithmic parameter
-  \gamma   - algorithmic parameter

   Options Database:
+  -ts_alpha_alpha_m <alpha_m> - set alpha_m
.  -ts_alpha_alpha_f <alpha_f> - set alpha_f
-  -ts_alpha_gamma   <gamma> - set gamma

  Note:
  Use of this function is normally only required to hack TSALPHA to
  use a modified integration scheme. Users should call
  TSAlphaSetRadius() to set the desired spectral radius of the methods
  (i.e. high-frequency damping) in order so select optimal values for
  these parameters.

  Level: advanced

.seealso: TSAlphaSetRadius(), TSAlphaGetParams()
@*/
PetscErrorCode TSAlphaSetParams(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,alpha_m,2);
  PetscValidLogicalCollectiveReal(ts,alpha_f,3);
  PetscValidLogicalCollectiveReal(ts,gamma,4);
  CHKERRQ(PetscTryMethod(ts,"TSAlphaSetParams_C",(TS,PetscReal,PetscReal,PetscReal),(ts,alpha_m,alpha_f,gamma)));
  PetscFunctionReturn(0);
}

/*@
  TSAlphaGetParams - gets the algorithmic parameters for TSALPHA

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameters:
+  \alpha_m - algorithmic parameter
.  \alpha_f - algorithmic parameter
-  \gamma   - algorithmic parameter

  Note:
  Use of this function is normally only required to hack TSALPHA to
  use a modified integration scheme. Users should call
  TSAlphaSetRadius() to set the high-frequency damping (i.e. spectral
  radius of the method) in order so select optimal values for these
  parameters.

  Level: advanced

.seealso: TSAlphaSetRadius(), TSAlphaSetParams()
@*/
PetscErrorCode TSAlphaGetParams(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (alpha_m) PetscValidRealPointer(alpha_m,2);
  if (alpha_f) PetscValidRealPointer(alpha_f,3);
  if (gamma)   PetscValidRealPointer(gamma,4);
  CHKERRQ(PetscUseMethod(ts,"TSAlphaGetParams_C",(TS,PetscReal*,PetscReal*,PetscReal*),(ts,alpha_m,alpha_f,gamma)));
  PetscFunctionReturn(0);
}
