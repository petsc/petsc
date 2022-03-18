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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* V1 = 1/(Gamma*dT)*(X1-X0) + (1-1/Gamma)*V0 */
  ierr = VecWAXPY(V1,-1.0,X0,X1);CHKERRQ(ierr);
  ierr = VecAXPBY(V1,1-1/Gamma,1/(Gamma*dt),V0);CHKERRQ(ierr);
  /* Xa = X0 + Alpha_f*(X1-X0) */
  ierr = VecWAXPY(Xa,-1.0,X0,X1);CHKERRQ(ierr);
  ierr = VecAYPX(Xa,Alpha_f,X0);CHKERRQ(ierr);
  /* Va = V0 + Alpha_m*(V1-V0) */
  ierr = VecWAXPY(Va,-1.0,V0,V1);CHKERRQ(ierr);
  ierr = VecAYPX(Va,Alpha_m,V0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlpha_SNESSolve(TS ts,Vec b,Vec x)
{
  PetscInt       nits,lits;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSolve(ts->snes,b,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(ts->snes,&nits);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(X0,&X1);CHKERRQ(ierr);

  /* Setup backward Euler with halved time step */
  ierr = TSAlphaGetParams(ts,&alpha_m,&alpha_f,&gamma);CHKERRQ(ierr);
  ierr = TSAlphaSetParams(ts,1,1,1);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&time_step);CHKERRQ(ierr);
  ts->time_step = time_step/2;
  ierr = TSAlpha_StageTime(ts);CHKERRQ(ierr);
  th->stage_time = ts->ptime;
  ierr = VecZeroEntries(th->V0);CHKERRQ(ierr);

  /* First BE step, (t0,X0) -> (t1,X1) */
  th->stage_time += ts->time_step;
  ierr = VecCopy(X0,th->X0);CHKERRQ(ierr);
  ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
  ierr = VecCopy(th->X0,X1);CHKERRQ(ierr);
  ierr = TSAlpha_SNESSolve(ts,NULL,X1);CHKERRQ(ierr);
  ierr = TSPostStage(ts,th->stage_time,0,&X1);CHKERRQ(ierr);
  ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X1,&stageok);CHKERRQ(ierr);
  if (!stageok) goto finally;

  /* Second BE step, (t1,X1) -> (t2,X2) */
  th->stage_time += ts->time_step;
  ierr = VecCopy(X1,th->X0);CHKERRQ(ierr);
  ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
  ierr = VecCopy(th->X0,X2);CHKERRQ(ierr);
  ierr = TSAlpha_SNESSolve(ts,NULL,X2);CHKERRQ(ierr);
  ierr = TSPostStage(ts,th->stage_time,0,&X2);CHKERRQ(ierr);
  ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X2,&stageok);CHKERRQ(ierr);
  if (!stageok) goto finally;

  /* Compute V0 ~ dX/dt at t0 with backward differences */
  ierr = VecZeroEntries(th->V0);CHKERRQ(ierr);
  ierr = VecAXPY(th->V0,-3/ts->time_step,X0);CHKERRQ(ierr);
  ierr = VecAXPY(th->V0,+4/ts->time_step,X1);CHKERRQ(ierr);
  ierr = VecAXPY(th->V0,-1/ts->time_step,X2);CHKERRQ(ierr);

  /* Rough, lower-order estimate LTE of the initial step */
  if (th->vec_lte_work) {
    ierr = VecZeroEntries(th->vec_lte_work);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_lte_work,+2,X2);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_lte_work,-4,X1);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_lte_work,+2,X0);CHKERRQ(ierr);
  }

 finally:
  /* Revert TSAlpha to the initial state (t0,X0) */
  if (initok) *initok = stageok;
  ierr = TSSetTimeStep(ts,time_step);CHKERRQ(ierr);
  ierr = TSAlphaSetParams(ts,alpha_m,alpha_f,gamma);CHKERRQ(ierr);
  ierr = VecCopy(ts->vec_sol,th->X0);CHKERRQ(ierr);

  ierr = VecDestroy(&X1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscInt       rejections = 0;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);

  if (!ts->steprollback) {
    if (th->vec_sol_prev) { ierr = VecCopy(th->X0,th->vec_sol_prev);CHKERRQ(ierr); }
    ierr = VecCopy(ts->vec_sol,th->X0);CHKERRQ(ierr);
    ierr = VecCopy(th->V1,th->V0);CHKERRQ(ierr);
  }

  th->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && th->status != TS_STEP_COMPLETE) {

    if (ts->steprestart) {
      ierr = TSAlpha_Restart(ts,&stageok);CHKERRQ(ierr);
      if (!stageok) goto reject_step;
    }

    ierr = TSAlpha_StageTime(ts);CHKERRQ(ierr);
    ierr = VecCopy(th->X0,th->X1);CHKERRQ(ierr);
    ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
    ierr = TSAlpha_SNESSolve(ts,NULL,th->X1);CHKERRQ(ierr);
    ierr = TSPostStage(ts,th->stage_time,0,&th->Xa);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,th->Xa,&stageok);CHKERRQ(ierr);
    if (!stageok) goto reject_step;

    th->status = TS_STEP_PENDING;
    ierr = VecCopy(th->X1,ts->vec_sol);CHKERRQ(ierr);
    ierr = TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
    th->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      ierr = VecCopy(th->X0,ts->vec_sol);CHKERRQ(ierr);
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
      ierr = PetscInfo(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->vec_sol_prev) {*wlte = -1; PetscFunctionReturn(0);}
  if (!th->vec_lte_work) {*wlte = -1; PetscFunctionReturn(0);}
  if (ts->steprestart) {
    /* th->vec_lte_work is set to the LTE in TSAlpha_Restart() */
    ierr = VecAXPY(Y,1,X);CHKERRQ(ierr);
  } else {
    /* Compute LTE using backward differences with non-constant time step */
    PetscReal   h = ts->time_step, h_prev = ts->ptime - ts->ptime_prev;
    PetscReal   a = 1 + h_prev/h;
    PetscScalar scal[3]; Vec vecs[3];
    scal[0] = +1/a;   scal[1] = -1/(a-1); scal[2] = +1/(a*(a-1));
    vecs[0] = th->X1; vecs[1] = th->X0;   vecs[2] = th->vec_sol_prev;
    ierr = VecCopy(X,Y);CHKERRQ(ierr);
    ierr = VecMAXPY(Y,3,scal,vecs);CHKERRQ(ierr);
  }
  ierr = TSErrorWeightedNorm(ts,X,Y,wnormtype,wlte,&wltea,&wlter);CHKERRQ(ierr);
  if (order) *order = 2;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(th->X0,ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_Alpha(TS ts,PetscReal t,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      dt  = t - ts->ptime;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,th->Gamma*dt,th->V1);CHKERRQ(ierr);
  ierr = VecAXPY(X,(1-th->Gamma)*dt,th->V0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormFunction_Alpha(PETSC_UNUSED SNES snes,Vec X,Vec F,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAlpha_StageVecs(ts,X);CHKERRQ(ierr);
  /* F = Function(ta,Xa,Va) */
  ierr = TSComputeIFunction(ts,ta,Xa,Va,F,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecScale(F,th->scale_F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_Alpha(PETSC_UNUSED SNES snes,PETSC_UNUSED Vec X,Mat J,Mat P,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va;
  PetscReal      dVdX = th->shift_V;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* J,P = Jacobian(ta,Xa,Va) */
  ierr = TSComputeIJacobian(ts,ta,Xa,Va,dVdX,J,P,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&th->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Xa);CHKERRQ(ierr);
  ierr = VecDestroy(&th->X1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Va);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol_prev);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_lte_work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Alpha(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Alpha(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetRadius_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetParams_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaGetParams_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&th->X0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xa);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->X1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Va);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V1);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&match);CHKERRQ(ierr);
  if (!match) {
    ierr = VecDuplicate(ts->vec_sol,&th->vec_sol_prev);CHKERRQ(ierr);
    ierr = VecDuplicate(ts->vec_sol,&th->vec_lte_work);CHKERRQ(ierr);
  }

  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_Alpha(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Generalized-Alpha ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flg;
    PetscReal radius = 1;
    ierr = PetscOptionsReal("-ts_alpha_radius","Spectral radius (high-frequency dissipation)","TSAlphaSetRadius",radius,&radius,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSAlphaSetRadius(ts,radius);CHKERRQ(ierr);}
    ierr = PetscOptionsReal("-ts_alpha_alpha_m","Algorithmic parameter alpha_m","TSAlphaSetParams",th->Alpha_m,&th->Alpha_m,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_alpha_f","Algorithmic parameter alpha_f","TSAlphaSetParams",th->Alpha_f,&th->Alpha_f,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_gamma","Algorithmic parameter gamma","TSAlphaSetParams",th->Gamma,&th->Gamma,NULL);CHKERRQ(ierr);
    ierr = TSAlphaSetParams(ts,th->Alpha_m,th->Alpha_f,th->Gamma);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Alpha(TS ts,PetscViewer viewer)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Alpha_m=%g, Alpha_f=%g, Gamma=%g\n",(double)th->Alpha_m,(double)th->Alpha_f,(double)th->Gamma);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAlphaSetRadius_Alpha(TS ts,PetscReal radius)
{
  PetscReal      alpha_m,alpha_f,gamma;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(radius < 0 || radius > 1,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  alpha_m = (PetscReal)0.5*(3-radius)/(1+radius);
  alpha_f = 1/(1+radius);
  gamma   = (PetscReal)0.5 + alpha_m - alpha_f;
  ierr = TSAlphaSetParams(ts,alpha_m,alpha_f,gamma);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

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

  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->Alpha_m = 0.5;
  th->Alpha_f = 0.5;
  th->Gamma   = 0.5;
  th->order   = 2;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetRadius_C",TSAlphaSetRadius_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetParams_C",TSAlphaSetParams_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaGetParams_C",TSAlphaGetParams_Alpha);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,radius,2);
  PetscCheckFalse(radius < 0 || radius > 1,((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  ierr = PetscTryMethod(ts,"TSAlphaSetRadius_C",(TS,PetscReal),(ts,radius));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,alpha_m,2);
  PetscValidLogicalCollectiveReal(ts,alpha_f,3);
  PetscValidLogicalCollectiveReal(ts,gamma,4);
  ierr = PetscTryMethod(ts,"TSAlphaSetParams_C",(TS,PetscReal,PetscReal,PetscReal),(ts,alpha_m,alpha_f,gamma));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (alpha_m) PetscValidRealPointer(alpha_m,2);
  if (alpha_f) PetscValidRealPointer(alpha_f,3);
  if (gamma)   PetscValidRealPointer(gamma,4);
  ierr = PetscUseMethod(ts,"TSAlphaGetParams_C",(TS,PetscReal*,PetscReal*,PetscReal*),(ts,alpha_m,alpha_f,gamma));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
