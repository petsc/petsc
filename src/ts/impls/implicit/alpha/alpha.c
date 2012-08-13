/*
  Code for timestepping with implicit generalized-\alpha method
  for first order systems.
*/
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef PetscErrorCode (*TSAlphaAdaptFunction)(TS,PetscReal,Vec,Vec,PetscReal*,PetscBool*,void*);

typedef struct {
  Vec X0,Xa,X1;
  Vec V0,Va,V1;
  Vec R,E;
  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Gamma;
  PetscReal stage_time;
  PetscReal shift;

  TSAlphaAdaptFunction adapt;
  void *adaptctx;
  PetscReal rtol;
  PetscReal atol;
  PetscReal rho;
  PetscReal scale_min;
  PetscReal scale_max;
  PetscReal dt_min;
  PetscReal dt_max;
} TS_Alpha;

#undef __FUNCT__
#define __FUNCT__ "TSStep_Alpha"
static PetscErrorCode TSStep_Alpha(TS ts)
{
  TS_Alpha            *th    = (TS_Alpha*)ts->data;
  PetscInt            its,lits,reject;
  PetscReal           next_time_step;
  SNESConvergedReason snesreason = SNES_CONVERGED_ITERATING;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (ts->steps == 0) {
    ierr = VecSet(th->V0,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(th->V1,th->V0);CHKERRQ(ierr);
  }
  ierr = VecCopy(ts->vec_sol,th->X0);CHKERRQ(ierr);
  next_time_step = ts->time_step;
  for (reject=0; reject<ts->max_reject; reject++,ts->reject++) {
    ts->time_step = next_time_step;
    th->stage_time = ts->ptime + th->Alpha_f*ts->time_step;
    th->shift = th->Alpha_m/(th->Alpha_f*th->Gamma*ts->time_step);
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
    /* predictor */
    ierr = VecCopy(th->X0,th->X1);CHKERRQ(ierr);
    /* solve R(X,V) = 0 */
    ierr = SNESSolve(ts->snes,PETSC_NULL,th->X1);CHKERRQ(ierr);
    /* V1 = (1-1/Gamma)*V0 + 1/(Gamma*dT)*(X1-X0) */
    ierr = VecWAXPY(th->V1,-1,th->X0,th->X1);CHKERRQ(ierr);
    ierr = VecAXPBY(th->V1,1-1/th->Gamma,1/(th->Gamma*ts->time_step),th->V0);CHKERRQ(ierr);
    /* nonlinear solve convergence */
    ierr = SNESGetConvergedReason(ts->snes,&snesreason);CHKERRQ(ierr);
    if (snesreason < 0 && !th->adapt) break;
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->snes_its += its; ts->ksp_its += lits;
    ierr = PetscInfo3(ts,"step=%D, nonlinear solve iterations=%D, linear solve iterations=%D\n",ts->steps,its,lits);CHKERRQ(ierr);
    /* time step adaptativity */
    if (!th->adapt) break;
    else {
      PetscReal t1 = ts->ptime + ts->time_step;
      PetscBool stepok = (reject==0) ? PETSC_TRUE : PETSC_FALSE;
      ierr = th->adapt(ts,t1,th->X1,th->V1,&next_time_step,&stepok,th->adaptctx);CHKERRQ(ierr);
      ierr = PetscInfo5(ts,"Step %D (t=%G,dt=%G) %s, next dt=%G\n",ts->steps,ts->ptime,ts->time_step,stepok?"accepted":"rejected",next_time_step);CHKERRQ(ierr);
      if (stepok) break;
    }
  }
  if (snesreason < 0 && ts->max_snes_failures > 0 && ++ts->num_snes_failures >= ts->max_snes_failures) {
    ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
    ierr = PetscInfo2(ts,"Step=%D, nonlinear solve solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (reject >= ts->max_reject) {
    ts->reason = TS_DIVERGED_STEP_REJECTED;
    ierr = PetscInfo2(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,reject);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecCopy(th->X1,ts->vec_sol);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  ts->steps++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_Alpha"
static PetscErrorCode TSInterpolate_Alpha(TS ts,PetscReal t,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      dt = t - ts->ptime;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,th->Gamma*dt,th->V1);CHKERRQ(ierr);
  ierr = VecAXPY(X,(1-th->Gamma)*dt,th->V0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "TSReset_Alpha"
static PetscErrorCode TSReset_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&th->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Xa);CHKERRQ(ierr);
  ierr = VecDestroy(&th->X1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Va);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->E);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_Alpha"
static PetscErrorCode TSDestroy_Alpha(TS ts)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSReset_Alpha(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaSetRadius_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaSetParams_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaGetParams_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaSetAdapt_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_Alpha"
static PetscErrorCode SNESTSFormFunction_Alpha(SNES snes,Vec x,Vec y,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  Vec            X0 = th->X0, V0 = th->V0;
  Vec            X1 = x, V1 = th->V1, R = y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* V1 = (1-1/Gamma)*V0 + 1/(Gamma*dT)*(X1-X0) */
  ierr = VecWAXPY(V1,-1,X0,X1);CHKERRQ(ierr);
  ierr = VecAXPBY(V1,1-1/th->Gamma,1/(th->Gamma*ts->time_step),V0);CHKERRQ(ierr);
  /* Xa = X0 + Alpha_f*(X1-X0) */
  ierr = VecWAXPY(th->Xa,-1,X0,X1);CHKERRQ(ierr);
  ierr = VecAYPX(th->Xa,th->Alpha_f,X0);CHKERRQ(ierr);
  /* Va = V0 + Alpha_m*(V1-V0) */
  ierr = VecWAXPY(th->Va,-1,V0,V1);CHKERRQ(ierr);
  ierr = VecAYPX(th->Va,th->Alpha_m,V0);CHKERRQ(ierr);
  /* F = Function(ta,Xa,Va) */
  ierr = TSComputeIFunction(ts,th->stage_time,th->Xa,th->Va,R,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecScale(R,1/th->Alpha_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_Alpha"
static PetscErrorCode SNESTSFormJacobian_Alpha(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *str,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* A,B = Jacobian(ta,Xa,Va) */
  ierr = TSComputeIJacobian(ts,th->stage_time,th->Xa,th->Va,th->shift,A,B,str,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Alpha"
static PetscErrorCode TSSetUp_Alpha(TS ts)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&th->X0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xa);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->X1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Va);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Alpha"
static PetscErrorCode TSSetFromOptions_Alpha(TS ts)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Alpha ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flag, adapt = PETSC_FALSE;
    PetscReal radius = 1.0;
    ierr = PetscOptionsReal("-ts_alpha_radius","spectral radius","TSAlphaSetRadius",radius,&radius,&flag);CHKERRQ(ierr);
    if (flag) { ierr = TSAlphaSetRadius(ts,radius);CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-ts_alpha_alpha_m","algoritmic parameter alpha_m","TSAlphaSetParams",th->Alpha_m,&th->Alpha_m,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_alpha_f","algoritmic parameter alpha_f","TSAlphaSetParams",th->Alpha_f,&th->Alpha_f,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_gamma","algoritmic parameter gamma","TSAlphaSetParams",th->Gamma,&th->Gamma,PETSC_NULL);CHKERRQ(ierr);
    ierr = TSAlphaSetParams(ts,th->Alpha_m,th->Alpha_f,th->Gamma);CHKERRQ(ierr);

    ierr = PetscOptionsBool("-ts_alpha_adapt","default time step adaptativity","TSAlphaSetAdapt",adapt,&adapt,&flag);CHKERRQ(ierr);
    if (flag) { ierr = TSAlphaSetAdapt(ts,adapt?TSAlphaAdaptDefault:PETSC_NULL,PETSC_NULL); CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-ts_alpha_adapt_rtol","relative tolerance for dt adaptativity","",th->rtol,&th->rtol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_adapt_atol","absolute tolerance for dt adaptativity","",th->atol,&th->atol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_adapt_min","minimum dt scale","",th->scale_min,&th->scale_min,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_adapt_max","maximum dt scale","",th->scale_max,&th->scale_max,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_adapt_dt_min","minimum dt","",th->dt_min,&th->dt_min,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_adapt_dt_max","maximum dt","",th->dt_max,&th->dt_max,PETSC_NULL);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_Alpha"
static PetscErrorCode TSView_Alpha(TS ts,PetscViewer viewer)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool       iascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Alpha_m=%G, Alpha_f=%G, Gamma=%G\n",th->Alpha_m,th->Alpha_f,th->Gamma);CHKERRQ(ierr);
  }
  ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetRadius_Alpha"
PetscErrorCode  TSAlphaSetRadius_Alpha(TS ts,PetscReal radius)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (radius < 0 || radius > 1) SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Radius %G not in range [0,1]",radius);
  th->Alpha_m = 0.5*(3-radius)/(1+radius);
  th->Alpha_f = 1/(1+radius);
  th->Gamma   = 0.5 + th->Alpha_m - th->Alpha_f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetParams_Alpha"
PetscErrorCode  TSAlphaSetParams_Alpha(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  th->Alpha_m = alpha_m;
  th->Alpha_f = alpha_f;
  th->Gamma   = gamma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaGetParams_Alpha"
PetscErrorCode  TSAlphaGetParams_Alpha(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (alpha_m) *alpha_m = th->Alpha_m;
  if (alpha_f) *alpha_f = th->Alpha_f;
  if (gamma)   *gamma   = th->Gamma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetAdapt_Alpha"
PetscErrorCode  TSAlphaSetAdapt_Alpha(TS ts,TSAlphaAdaptFunction adapt,void *ctx)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  th->adapt    = adapt;
  th->adaptctx = ctx;
  PetscFunctionReturn(0);
}

EXTERN_C_END

/* ------------------------------------------------------------ */
/*MC
      TSALPHA - DAE solver using the implicit Generalized-Alpha method

  Level: beginner

  References:
  K.E. Jansen, C.H. Whiting, G.M. Hulber, "A generalized-alpha
  method for integrating the filtered Navier-Stokes equations with a
  stabilized finite element method", Computer Methods in Applied
  Mechanics and Engineering, 190, 305-319, 2000.
  DOI: 10.1016/S0045-7825(00)00203-6.

  J. Chung, G.M.Hubert. "A Time Integration Algorithm for Structural
  Dynamics with Improved Numerical Dissipation: The Generalized-alpha
  Method" ASME Journal of Applied Mechanics, 60, 371:375, 1993.

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Alpha"
PetscErrorCode  TSCreate_Alpha(TS ts)
{
  TS_Alpha       *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Alpha;
  ts->ops->destroy        = TSDestroy_Alpha;
  ts->ops->view           = TSView_Alpha;
  ts->ops->setup          = TSSetUp_Alpha;
  ts->ops->step           = TSStep_Alpha;
  ts->ops->interpolate    = TSInterpolate_Alpha;
  ts->ops->setfromoptions = TSSetFromOptions_Alpha;
  ts->ops->snesfunction   = SNESTSFormFunction_Alpha;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Alpha;

  ierr = PetscNewLog(ts,TS_Alpha,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->Alpha_m = 0.5;
  th->Alpha_f = 0.5;
  th->Gamma   = 0.5;

  th->rtol      = 1e-3;
  th->atol      = 1e-3;
  th->rho       = 0.9;
  th->scale_min = 0.1;
  th->scale_max = 5.0;
  th->dt_min    = 0.0;
  th->dt_max    = PETSC_MAX_REAL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaSetAdapt_C","TSAlphaSetAdapt_Alpha",TSAlphaSetAdapt_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaSetRadius_C","TSAlphaSetRadius_Alpha",TSAlphaSetRadius_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaSetParams_C","TSAlphaSetParams_Alpha",TSAlphaSetParams_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlphaGetParams_C","TSAlphaGetParams_Alpha",TSAlphaGetParams_Alpha);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetAdapt"
/*@C
  TSAlphaSetAdapt - sets the time step adaptativity and acceptance test routine

  This function allows to accept/reject a step and select the
  next time step to use.

  Not Collective

  Input Parameter:
+  ts - timestepping context
.  adapt - user-defined adapt routine
-  ctx  - [optional] user-defined context for private data for the
         adapt routine (may be PETSC_NULL)

   Calling sequence of adapt:
$    adapt (TS ts,PetscReal t,Vec X,Vec Xdot,
$            PetscReal *next_dt,PetscBool *accepted,void *ctx);

  Level: intermediate

@*/
PetscErrorCode  TSAlphaSetAdapt(TS ts,TSAlphaAdaptFunction adapt,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSAlphaSetAdapt_C",(TS,TSAlphaAdaptFunction,void*),(ts,adapt,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaAdaptDefault"
PetscErrorCode  TSAlphaAdaptDefault(TS ts,PetscReal t,Vec X,Vec Xdot, PetscReal *nextdt,PetscBool *ok,void *ctx)
{
  TS_Alpha            *th;
  SNESConvergedReason snesreason;
  PetscReal           dt,normX,normE,Emax,scale;
  PetscErrorCode      ierr;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
#if PETSC_USE_DEBUG
  {
    PetscBool match;
    ierr = PetscObjectTypeCompare((PetscObject)ts,TSALPHA,&match);CHKERRQ(ierr);
    if (!match) SETERRQ(((PetscObject)ts)->comm,1,"Only for TSALPHA");
  }
#endif
  th = (TS_Alpha*)ts->data;

  ierr = SNESGetConvergedReason(ts->snes,&snesreason);CHKERRQ(ierr);
  if (snesreason < 0) {
    *ok = PETSC_FALSE;
    *nextdt *= th->scale_min;
    goto finally;
  }

  /* first-order aproximation to the local error */
  /* E = (X0 + dt*Xdot) - X */
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (!th->E) {ierr = VecDuplicate(th->X0,&th->E);CHKERRQ(ierr);}
  ierr = VecWAXPY(th->E,dt,Xdot,th->X0);CHKERRQ(ierr);
  ierr = VecAXPY(th->E,-1,X);CHKERRQ(ierr);
  ierr = VecNorm(th->E,NORM_2,&normE);CHKERRQ(ierr);
  /* compute maximum allowable error */
  ierr = VecNorm(X,NORM_2,&normX);CHKERRQ(ierr);
  if (normX == 0) {ierr = VecNorm(th->X0,NORM_2,&normX);CHKERRQ(ierr);}
  Emax =  th->rtol * normX + th->atol;
  /* compute next time step */
  if (normE > 0) {
    scale = th->rho * PetscRealPart(PetscSqrtScalar((PetscScalar)(Emax/normE)));
    scale = PetscMax(scale,th->scale_min);
    scale = PetscMin(scale,th->scale_max);
    if (!(*ok))
      scale = PetscMin(1.0,scale);
    *nextdt *= scale;
  }
  /* accept or reject step */
  if (normE <= Emax)
    *ok = PETSC_TRUE;
  else
    *ok = PETSC_FALSE;

  finally:
  *nextdt = PetscMax(*nextdt,th->dt_min);
  *nextdt = PetscMin(*nextdt,th->dt_max);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetRadius"
/*@
  TSAlphaSetRadius - sets the desired spectral radius of the method
                     (i.e. high-frequency numerical damping)

  Logically Collective on TS

  The algorithmic parameters \alpha_m and \alpha_f of the
  generalized-\alpha method can be computed in terms of a specified
  spectral radius \rho in [0,1] for infinite time step in order to
  control high-frequency numerical damping:
    alpha_m = 0.5*(3-\rho)/(1+\rho)
    alpha_f = 1/(1+\rho)

  Input Parameter:
+  ts - timestepping context
-  radius - the desired spectral radius

  Options Database:
.  -ts_alpha_radius <radius>

  Level: intermediate

.seealso: TSAlphaSetParams(), TSAlphaGetParams()
@*/
PetscErrorCode  TSAlphaSetRadius(TS ts,PetscReal radius)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSAlphaSetRadius_C",(TS,PetscReal),(ts,radius));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetParams"
/*@
  TSAlphaSetParams - sets the algorithmic parameters for TSALPHA

  Not Collective

  Second-order accuracy can be obtained so long as:
    \gamma = 0.5 + alpha_m - alpha_f

  Unconditional stability requires:
    \alpha_m >= \alpha_f >= 0.5

  Backward Euler method is recovered when:
    \alpha_m = \alpha_f = gamma = 1


  Input Parameter:
+  ts - timestepping context
.  \alpha_m - algorithmic paramenter
.  \alpha_f - algorithmic paramenter
-  \gamma   - algorithmic paramenter

   Options Database:
+  -ts_alpha_alpha_m <alpha_m>
.  -ts_alpha_alpha_f <alpha_f>
-  -ts_alpha_gamma <gamma>

  Note:
  Use of this function is normally only required to hack TSALPHA to
  use a modified integration scheme. Users should call
  TSAlphaSetRadius() to set the desired spectral radius of the methods
  (i.e. high-frequency damping) in order so select optimal values for
  these parameters.

  Level: advanced

.seealso: TSAlphaSetRadius(), TSAlphaGetParams()
@*/
PetscErrorCode  TSAlphaSetParams(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSAlphaSetParams_C",(TS,PetscReal,PetscReal,PetscReal),(ts,alpha_m,alpha_f,gamma));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaGetParams"
/*@
  TSAlphaGetParams - gets the algorithmic parameters for TSALPHA

  Not Collective

  Input Parameter:
+  ts - timestepping context
.  \alpha_m - algorithmic parameter
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
PetscErrorCode  TSAlphaGetParams(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (alpha_m) PetscValidPointer(alpha_m,2);
  if (alpha_f) PetscValidPointer(alpha_f,3);
  if (gamma)   PetscValidPointer(gamma,4);
  ierr = PetscUseMethod(ts,"TSAlphaGetParams_C",(TS,PetscReal*,PetscReal*,PetscReal*),(ts,alpha_m,alpha_f,gamma));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
