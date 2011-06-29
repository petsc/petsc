/*
  Code for timestepping with implicit Theta method

  Notes:
  This method can be applied to DAE.

  This method is cast as a 1-stage implicit Runge-Kutta method.

  Theta | Theta
  -------------
        |  1

  To apply a diagonally implicit RK method to DAE, the stage formula

  X_i = x + h sum_j a_ij X'_j

  is interpreted as a formula for X'_i in terms of X_i and known stuff (X'_j, j<i)
*/
#include <private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef struct {
  Vec       X,Xdot;                   /* Storage for one stage */
  PetscBool extrapolate;
  PetscReal Theta;
  PetscReal shift;
  PetscReal stage_time;
} TS_Theta;

#undef __FUNCT__
#define __FUNCT__ "TSStep_Theta"
static PetscErrorCode TSStep_Theta(TS ts,PetscInt *steps,PetscReal *ptime)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscInt       i,its,lits;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *steps = -ts->steps;
  *ptime =  ts->ptime;

  ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  for (i=0; i<ts->max_steps; i++) {
    if (ts->ptime + ts->time_step > ts->max_time) break;
    ierr = TSPreStep(ts);CHKERRQ(ierr);

    th->stage_time = ts->ptime + th->Theta*ts->time_step;
    th->shift = 1./(th->Theta*ts->time_step);

    if (th->extrapolate) {
      ierr = VecWAXPY(th->X,1./th->shift,th->Xdot,ts->vec_sol);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(ts->vec_sol,th->X);CHKERRQ(ierr);
    }
    ierr = SNESSolve(ts->snes,PETSC_NULL,th->X);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;

    ierr = VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,ts->vec_sol,th->X);CHKERRQ(ierr);
    ierr = VecAXPY(ts->vec_sol,ts->time_step,th->Xdot);CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ts->steps++;

    ierr = TSPostStep(ts);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
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

  PetscFunctionBegin;
  ierr = VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,ts->vec_sol,x);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,th->stage_time,x,th->Xdot,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_Theta"
static PetscErrorCode SNESTSFormJacobian_Theta(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *str,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* th->Xdot has already been computed in SNESTSFormFunction_Theta (SNES guarantees this) */
  ierr = TSComputeIJacobian(ts,th->stage_time,x,th->Xdot,th->shift,A,B,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Theta"
static PetscErrorCode TSSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ts->problem_type == TS_LINEAR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Only for nonlinear problems");
  ierr = VecDuplicate(ts->vec_sol,&th->X);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xdot);CHKERRQ(ierr);
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
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Theta=%G\n",th->Theta);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Extrapolation=%s\n",th->extrapolate?"yes":"no");CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for TS_Theta",((PetscObject)viewer)->type_name);
  }
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
EXTERN_C_END

/* ------------------------------------------------------------ */
/*MC
      TSTHETA - DAE solver using the implicit Theta method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Theta"
PetscErrorCode  TSCreate_Theta(TS ts)
{
  TS_Theta       *th;
  PetscErrorCode ierr;
  SNES           snes;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Theta;
  ts->ops->destroy        = TSDestroy_Theta;
  ts->ops->view           = TSView_Theta;
  ts->ops->setup          = TSSetUp_Theta;
  ts->ops->step           = TSStep_Theta;
  ts->ops->setfromoptions = TSSetFromOptions_Theta;
  ts->ops->snesfunction   = SNESTSFormFunction_Theta;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Theta;

  ts->problem_type = TS_NONLINEAR;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);

  ierr = PetscNewLog(ts,TS_Theta,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->extrapolate = PETSC_FALSE;
  th->Theta       = 0.5;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaGetTheta_C","TSThetaGetTheta_Theta",TSThetaGetTheta_Theta);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSThetaSetTheta_C","TSThetaSetTheta_Theta",TSThetaSetTheta_Theta);CHKERRQ(ierr);
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
