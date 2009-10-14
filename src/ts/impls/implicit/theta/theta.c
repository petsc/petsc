#define PETSCTS_DLL

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
#include "private/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec Xold;
  Vec X,Xdot;                   /* Storage for one stage */
  Vec res;                      /* DAE residuals */
  PetscTruth extrapolate;
  PetscReal Theta;
  PetscReal shift;
  PetscReal stage_time;
} TS_Theta;

#undef __FUNCT__  
#define __FUNCT__ "TSStep_Theta"
static PetscErrorCode TSStep_Theta(TS ts,PetscInt *steps,PetscReal *ptime)
{
  Vec            sol = ts->vec_sol;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its,lits;
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    if (ts->ptime + ts->time_step > ts->max_time) break;
    th->stage_time = ts->ptime + th->Theta*ts->time_step;
    th->shift = 1./(th->Theta*ts->time_step);
    ts->ptime += ts->time_step;

    ierr = VecCopy(sol,th->Xold);CHKERRQ(ierr); /* Used within function evalutaion */
    if (th->extrapolate) {
      ierr = VecWAXPY(th->X,1./th->shift,th->Xdot,sol);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(sol,th->X);CHKERRQ(ierr);
    }
    ierr = SNESSolve(ts->snes,PETSC_NULL,th->X);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;
    ierr = VecAXPY(sol,ts->time_step,th->Xdot);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_Theta"
static PetscErrorCode TSDestroy_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(th->Xold);CHKERRQ(ierr);
  ierr = VecDestroy(th->X);CHKERRQ(ierr);
  ierr = VecDestroy(th->Xdot);CHKERRQ(ierr);
  ierr = VecDestroy(th->res);CHKERRQ(ierr);
  ierr = PetscFree(th);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    This defines the nonlinear equation that is to be solved with SNES
    G(U) = F[t0+T*dt, U, (U-U0)*shift] = 0
*/
#undef __FUNCT__  
#define __FUNCT__ "TSThetaFunction"
static PetscErrorCode TSThetaFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS        ts = (TS)ctx;
  TS_Theta *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,th->Xold,x);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,th->stage_time,x,th->Xdot,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSThetaJacobian"
static PetscErrorCode TSThetaJacobian(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *str,void *ctx)
{
  TS        ts = (TS)ctx;
  TS_Theta *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* th->Xdot will have already been computed in TSThetaFunction */
  ierr = TSComputeIJacobian(ts,th->stage_time,x,th->Xdot,th->shift,A,B,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_Theta"
static PetscErrorCode TSSetUp_Theta(TS ts)
{
  TS_Theta *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&th->Xold);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->X);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xdot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->res);CHKERRQ(ierr);
  ierr = SNESSetFunction(ts->snes,th->res,TSThetaFunction,ts);CHKERRQ(ierr);
  /* This is nasty.  SNESSetFromOptions() is usually called in TSSetFromOptions().  With -snes_mf_operator, it will
  replace A and we don't want to mess with that.  With -snes_mf, A and B will be replaced as well as the function and
  context.  Note that SNESSetFunction() normally has not been called before SNESSetFromOptions(), so when -snes_mf sets
  the Jacobian user context to snes->funP, it will actually be NULL.  This is not a problem because both snes->funP and
  snes->jacP should be the TS. */
  {
    Mat A,B;
    PetscErrorCode (*func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
    void *ctx;
    ierr = SNESGetJacobian(ts->snes,&A,&B,&func,&ctx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(ts->snes,A?A:ts->A,B?B:ts->B,func?func:&TSThetaJacobian,ctx?ctx:ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_Theta"
static PetscErrorCode TSSetFromOptions_Theta(TS ts)
{
  TS_Theta *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Theta ODE solver options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-ts_theta_theta","Location of stage (0<Theta<=1)","TSThetaSetTheta",th->Theta,&th->Theta,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ts_theta_extrapolate","Extrapolate stage solution from previous solution (sometimes unstable)","TSThetaSetExtrapolate",th->extrapolate,&th->extrapolate,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_Theta"
static PetscErrorCode TSView_Theta(TS ts,PetscViewer viewer)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscTruth      iascii;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Theta=%G\n",th->Theta);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Extrapolation=%s\n",th->extrapolate?"yes":"no");CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for TS_Theta",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSTHETA - DAE solver using the implicit Theta method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_Theta"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Theta(TS ts)
{
  TS_Theta       *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->destroy        = TSDestroy_Theta;
  ts->ops->view           = TSView_Theta;
  ts->ops->setup          = TSSetUp_Theta;
  ts->ops->step           = TSStep_Theta;
  ts->ops->setfromoptions = TSSetFromOptions_Theta;

  ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);

  ierr = PetscNewLog(ts,TS_Theta,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->extrapolate = PETSC_TRUE;
  th->Theta       = 0.5;

  PetscFunctionReturn(0);
}
EXTERN_C_END
