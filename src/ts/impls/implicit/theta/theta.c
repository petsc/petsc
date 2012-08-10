/*
  Code for timestepping with implicit Theta method
*/
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscsnesfas.h>

typedef struct {
  Vec       X,Xdot;                   /* Storage for one stage */
  Vec       affine;                   /* Affine vector needed for residual at beginning of step */
  PetscBool extrapolate;
  PetscBool endpoint;
  PetscReal Theta;
  PetscReal shift;
  PetscReal stage_time;
} TS_Theta;

#undef __FUNCT__
#define __FUNCT__ "TSThetaGetX0AndXdot"
static PetscErrorCode TSThetaGetX0AndXdot(TS ts,DM dm,Vec *X0,Vec *Xdot)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {
      ierr = PetscObjectQuery((PetscObject)dm,"TSTheta_X0",(PetscObject*)X0);CHKERRQ(ierr);
      if (!*X0) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_ARG_INCOMP,"TSTheta_X0 has not been composed with DM from SNES");
    } else *X0 = ts->vec_sol;
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {
      ierr = PetscObjectQuery((PetscObject)dm,"TSTheta_Xdot",(PetscObject*)Xdot);CHKERRQ(ierr);
      if (!*Xdot) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_ARG_INCOMP,"TSTheta_Xdot has not been composed with DM from SNES");
    } else *Xdot = th->Xdot;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_TSTheta"
static PetscErrorCode DMCoarsenHook_TSTheta(DM fine,DM coarse,void *ctx)
{
  Vec X0,Xdot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(coarse,&X0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(coarse,&Xdot);CHKERRQ(ierr);
  /* Oh noes, this would create a loop because the Vec holds a reference to the DM.
     Making a PetscContainer to hold these Vecs would make the following call succeed, but would create a reference loop.
     Need to decide on a way to break the reference counting loop.
   */
  ierr = PetscObjectCompose((PetscObject)coarse,"TSTheta_X0",(PetscObject)X0);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)coarse,"TSTheta_Xdot",(PetscObject)Xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&X0);CHKERRQ(ierr);
  ierr = VecDestroy(&Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrictHook_TSTheta"
static PetscErrorCode DMRestrictHook_TSTheta(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec X0,Xdot,X0_c,Xdot_c;

  PetscFunctionBegin;
  ierr = TSThetaGetX0AndXdot(ts,fine,&X0,&Xdot);CHKERRQ(ierr);
  ierr = TSThetaGetX0AndXdot(ts,coarse,&X0_c,&Xdot_c);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,X0,X0_c);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,Xdot,Xdot_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(X0_c,rscale,X0_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Xdot_c,rscale,Xdot_c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_Theta"
static PetscErrorCode TSStep_Theta(TS ts)
{
  TS_Theta            *th = (TS_Theta*)ts->data;
  PetscInt            its,lits;
  PetscReal           next_time_step;
  SNESConvergedReason snesreason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  next_time_step = ts->time_step;
  th->stage_time = ts->ptime + (th->endpoint ? 1. : th->Theta)*ts->time_step;
  th->shift = 1./(th->Theta*ts->time_step);
  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);

  if (th->endpoint) {           /* This formulation assumes linear time-independent mass matrix */
    ierr = VecZeroEntries(th->Xdot);CHKERRQ(ierr);
    if (!th->affine) {ierr = VecDuplicate(ts->vec_sol,&th->affine);CHKERRQ(ierr);}
    ierr = TSComputeIFunction(ts,ts->ptime,ts->vec_sol,th->Xdot,th->affine,PETSC_FALSE);CHKERRQ(ierr);
    ierr = VecScale(th->affine,(th->Theta-1.)/th->Theta);CHKERRQ(ierr);
  }
  if (th->extrapolate) {
    ierr = VecWAXPY(th->X,1./th->shift,th->Xdot,ts->vec_sol);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(ts->vec_sol,th->X);CHKERRQ(ierr);
  }
  ierr = SNESSolve(ts->snes,th->affine,th->X);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(ts->snes,&snesreason);CHKERRQ(ierr);
  ts->snes_its += its; ts->ksp_its += lits;
  if (snesreason < 0 && ts->max_snes_failures > 0 && ++ts->num_snes_failures >= ts->max_snes_failures) {
    ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
    ierr = PetscInfo2(ts,"Step=%D, nonlinear solve solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (th->endpoint) {
    ierr = VecCopy(th->X,ts->vec_sol);CHKERRQ(ierr);
  } else {
    ierr = VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,ts->vec_sol,th->X);CHKERRQ(ierr);
    ierr = VecAXPY(ts->vec_sol,ts->time_step,th->Xdot);CHKERRQ(ierr);
  }
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  ts->steps++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_Theta"
static PetscErrorCode TSInterpolate_Theta(TS ts,PetscReal t,Vec X)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscReal      alpha = t - ts->ptime;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,th->X);CHKERRQ(ierr);
  if (th->endpoint) alpha *= th->Theta;
  ierr = VecWAXPY(X,alpha,th->Xdot,th->X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "TSReset_Theta"
static PetscErrorCode TSReset_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&th->X);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&th->affine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_Theta"
static PetscErrorCode TSDestroy_Theta(TS ts)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSReset_Theta(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaGetTheta_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaSetTheta_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaGetEndpoint_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaSetEndpoint_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+Theta*dt, U, (U-U0)*shift] = 0
*/
#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_Theta"
static PetscErrorCode SNESTSFormFunction_Theta(SNES snes,Vec x,Vec y,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;
  Vec            X0,Xdot;
  DM             dm,dmsave;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  /* When using the endpoint variant, this is actually 1/Theta * Xdot */
  ierr = TSThetaGetX0AndXdot(ts,dm,&X0,&Xdot);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(Xdot,-th->shift,th->shift,0,X0,x);CHKERRQ(ierr);

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr = TSComputeIFunction(ts,th->stage_time,x,Xdot,y,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_Theta"
static PetscErrorCode SNESTSFormJacobian_Theta(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *str,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;
  Vec            Xdot;
  DM             dm,dmsave;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);

  /* th->Xdot has already been computed in SNESTSFormFunction_Theta (SNES guarantees this) */
  ierr = TSThetaGetX0AndXdot(ts,dm,PETSC_NULL,&Xdot);CHKERRQ(ierr);

  dmsave = ts->dm;
  ts->dm = dm;
  ierr = TSComputeIJacobian(ts,th->stage_time,x,Xdot,th->shift,A,B,str,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Theta"
static PetscErrorCode TSSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;
  SNES           snes;
  DM             dm;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&th->X);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xdot);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  if (dm) {
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSTheta,DMRestrictHook_TSTheta,ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Theta"
static PetscErrorCode TSSetFromOptions_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Theta ODE solver options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-ts_theta_theta","Location of stage (0<Theta<=1)","TSThetaSetTheta",th->Theta,&th->Theta,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_theta_extrapolate","Extrapolate stage solution from previous solution (sometimes unstable)","TSThetaSetExtrapolate",th->extrapolate,&th->extrapolate,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_theta_endpoint","Use the endpoint instead of midpoint form of the Theta method","TSThetaSetEndpoint",th->endpoint,&th->endpoint,PETSC_NULL);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_Theta"
static PetscErrorCode TSView_Theta(TS ts,PetscViewer viewer)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscBool       iascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Theta=%G\n",th->Theta);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Extrapolation=%s\n",th->extrapolate?"yes":"no");CHKERRQ(ierr);
  }
  ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSThetaGetTheta_Theta"
PetscErrorCode  TSThetaGetTheta_Theta(TS ts,PetscReal *theta)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  *theta = th->Theta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSThetaSetTheta_Theta"
PetscErrorCode  TSThetaSetTheta_Theta(TS ts,PetscReal theta)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  if (theta <= 0 || 1 < theta) SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Theta %G not in range (0,1]",theta);
  th->Theta = theta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSThetaGetEndpoint_Theta"
PetscErrorCode  TSThetaGetEndpoint_Theta(TS ts,PetscBool *endpoint)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  *endpoint = th->endpoint;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSThetaSetEndpoint_Theta"
PetscErrorCode  TSThetaSetEndpoint_Theta(TS ts,PetscBool flg)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  th->endpoint = flg;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------ */
/*MC
      TSTHETA - DAE solver using the implicit Theta method

   Level: beginner

   Notes:
   This method can be applied to DAE.

   This method is cast as a 1-stage implicit Runge-Kutta method.

.vb
  Theta | Theta
  -------------
        |  1
.ve

   For the default Theta=0.5, this is also known as the implicit midpoint rule.

   When the endpoint variant is chosen, the method becomes a 2-stage method with first stage explicit:

.vb
  0 | 0         0
  1 | 1-Theta   Theta
  -------------------
    | 1-Theta   Theta
.ve

   For the default Theta=0.5, this is the trapezoid rule (also known as Crank-Nicolson, see TSCN).

   To apply a diagonally implicit RK method to DAE, the stage formula

$  Y_i = X + h sum_j a_ij Y'_j

   is interpreted as a formula for Y'_i in terms of Y_i and known stuff (Y'_j, j<i)

.seealso:  TSCreate(), TS, TSSetType(), TSCN, TSBEULER, TSThetaSetTheta(), TSThetaSetEndpoint()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Theta"
PetscErrorCode  TSCreate_Theta(TS ts)
{
  TS_Theta       *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Theta;
  ts->ops->destroy        = TSDestroy_Theta;
  ts->ops->view           = TSView_Theta;
  ts->ops->setup          = TSSetUp_Theta;
  ts->ops->step           = TSStep_Theta;
  ts->ops->interpolate    = TSInterpolate_Theta;
  ts->ops->setfromoptions = TSSetFromOptions_Theta;
  ts->ops->snesfunction   = SNESTSFormFunction_Theta;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Theta;

  ierr = PetscNewLog(ts,TS_Theta,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->extrapolate = PETSC_FALSE;
  th->Theta       = 0.5;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaGetTheta_C","TSThetaGetTheta_Theta",TSThetaGetTheta_Theta);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaSetTheta_C","TSThetaSetTheta_Theta",TSThetaSetTheta_Theta);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaGetEndpoint_C","TSThetaGetEndpoint_Theta",TSThetaGetEndpoint_Theta);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaSetEndpoint_C","TSThetaSetEndpoint_Theta",TSThetaSetEndpoint_Theta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "TSThetaGetTheta"
/*@
  TSThetaGetTheta - Get the abscissa of the stage in (0,1].

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  theta - stage abscissa

  Note:
  Use of this function is normally only required to hack TSTHETA to use a modified integration scheme.

  Level: Advanced

.seealso: TSThetaSetTheta()
@*/
PetscErrorCode  TSThetaGetTheta(TS ts,PetscReal *theta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(theta,2);
  ierr = PetscUseMethod(ts,"TSThetaGetTheta_C",(TS,PetscReal*),(ts,theta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSThetaSetTheta"
/*@
  TSThetaSetTheta - Set the abscissa of the stage in (0,1].

  Not Collective

  Input Parameter:
+  ts - timestepping context
-  theta - stage abscissa

  Options Database:
.  -ts_theta_theta <theta>

  Level: Intermediate

.seealso: TSThetaGetTheta()
@*/
PetscErrorCode  TSThetaSetTheta(TS ts,PetscReal theta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSThetaSetTheta_C",(TS,PetscReal),(ts,theta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSThetaGetEndpoint"
/*@
  TSThetaGetEndpoint - Gets whether to use the endpoint variant of the method (e.g. trapezoid/Crank-Nicolson instead of midpoint rule).

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  endpoint - PETSC_TRUE when using the endpoint variant

  Level: Advanced

.seealso: TSThetaSetEndpoint(), TSTHETA, TSCN
@*/
PetscErrorCode TSThetaGetEndpoint(TS ts,PetscBool *endpoint)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(endpoint,2);
  ierr = PetscTryMethod(ts,"TSThetaGetEndpoint_C",(TS,PetscBool*),(ts,endpoint));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSThetaSetEndpoint"
/*@
  TSThetaSetEndpoint - Sets whether to use the endpoint variant of the method (e.g. trapezoid/Crank-Nicolson instead of midpoint rule).

  Not Collective

  Input Parameter:
+  ts - timestepping context
-  flg - PETSC_TRUE to use the endpoint variant

  Options Database:
.  -ts_theta_endpoint <flg>

  Level: Intermediate

.seealso: TSTHETA, TSCN
@*/
PetscErrorCode TSThetaSetEndpoint(TS ts,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSThetaSetEndpoint_C",(TS,PetscBool),(ts,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * TSBEULER and TSCN are straightforward specializations of TSTHETA.
 * The creation functions for these specializations are below.
 */

#undef __FUNCT__
#define __FUNCT__ "TSView_BEuler"
static PetscErrorCode TSView_BEuler(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSBEULER - ODE solver using the implicit backward Euler method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSEULER, TSCN, TSTHETA

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_BEuler"
PetscErrorCode  TSCreate_BEuler(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate_Theta(ts);CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts,1.0);CHKERRQ(ierr);
  ts->ops->view = TSView_BEuler;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "TSView_CN"
static PetscErrorCode TSView_CN(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSCN - ODE solver using the implicit Crank-Nicolson method.

  Level: beginner

  Notes:
  TSCN is equivalent to TSTHETA with Theta=0.5 and the "endpoint" option set. I.e.

$  -ts_type theta -ts_theta_theta 0.5 -ts_theta_endpoint

.seealso:  TSCreate(), TS, TSSetType(), TSBEULER, TSTHETA

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_CN"
PetscErrorCode  TSCreate_CN(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate_Theta(ts);CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts,0.5);CHKERRQ(ierr);
  ierr = TSThetaSetEndpoint(ts,PETSC_TRUE);CHKERRQ(ierr);
  ts->ops->view = TSView_CN;
  PetscFunctionReturn(0);
}
EXTERN_C_END
