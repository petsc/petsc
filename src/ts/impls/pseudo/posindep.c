#define PETSCTS_DLL

/*
       Code for Timestepping with implicit backwards Euler.
*/
#include "private/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  xdot;        /* work vector for time derivative of state */

  /* information used for Pseudo-timestepping */

  PetscErrorCode (*dt)(TS,PetscReal*,void*);              /* compute next timestep, and related context */
  void           *dtctx;              
  PetscErrorCode (*verify)(TS,Vec,void*,PetscReal*,PetscTruth*); /* verify previous timestep and related context */
  void           *verifyctx;     

  PetscReal  initial_fnorm,fnorm;                  /* original and current norm of F(u) */
  PetscReal  fnorm_previous;

  PetscReal  dt_increment;                  /* scaling that dt is incremented each time-step */
  PetscTruth increment_dt_from_initial_dt;
} TS_Pseudo;

/* ------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoComputeTimeStep"
/*@
    TSPseudoComputeTimeStep - Computes the next timestep for a currently running
    pseudo-timestepping process.

    Collective on TS

    Input Parameter:
.   ts - timestep context

    Output Parameter:
.   dt - newly computed timestep

    Level: advanced

    Notes:
    The routine to be called here to compute the timestep should be
    set by calling TSPseudoSetTimeStep().

.keywords: timestep, pseudo, compute

.seealso: TSPseudoDefaultTimeStep(), TSPseudoSetTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoComputeTimeStep(TS ts,PetscReal *dt)
{
  TS_Pseudo      *pseudo = (TS_Pseudo*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(TS_PseudoComputeTimeStep,ts,0,0,0);CHKERRQ(ierr);
  ierr = (*pseudo->dt)(ts,dt,pseudo->dtctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TS_PseudoComputeTimeStep,ts,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoDefaultVerifyTimeStep"
/*@C
   TSPseudoDefaultVerifyTimeStep - Default code to verify the quality of the last timestep.

   Collective on TS

   Input Parameters:
+  ts - the timestep context
.  dtctx - unused timestep context
-  update - latest solution vector

   Output Parameters:
+  newdt - the timestep to use for the next step
-  flag - flag indicating whether the last time step was acceptable

   Level: advanced

   Note:
   This routine always returns a flag of 1, indicating an acceptable 
   timestep.

.keywords: timestep, pseudo, default, verify 

.seealso: TSPseudoSetVerifyTimeStep(), TSPseudoVerifyTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoDefaultVerifyTimeStep(TS ts,Vec update,void *dtctx,PetscReal *newdt,PetscTruth *flag)
{
  PetscFunctionBegin;
  *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSPseudoVerifyTimeStep"
/*@
    TSPseudoVerifyTimeStep - Verifies whether the last timestep was acceptable.

    Collective on TS

    Input Parameters:
+   ts - timestep context
-   update - latest solution vector

    Output Parameters:
+   dt - newly computed timestep (if it had to shrink)
-   flag - indicates if current timestep was ok

    Level: advanced

    Notes:
    The routine to be called here to compute the timestep should be
    set by calling TSPseudoSetVerifyTimeStep().

.keywords: timestep, pseudo, verify 

.seealso: TSPseudoSetVerifyTimeStep(), TSPseudoDefaultVerifyTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoVerifyTimeStep(TS ts,Vec update,PetscReal *dt,PetscTruth *flag)
{
  TS_Pseudo      *pseudo = (TS_Pseudo*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pseudo->verify) {*flag = PETSC_TRUE; PetscFunctionReturn(0);}

  ierr = (*pseudo->verify)(ts,update,pseudo->verifyctx,dt,flag);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSStep_Pseudo"
static PetscErrorCode TSStep_Pseudo(TS ts,PetscInt *steps,PetscReal *ptime)
{
  Vec            sol = ts->vec_sol;
  PetscErrorCode ierr;
  PetscInt       i,max_steps = ts->max_steps,its,lits;
  PetscTruth     ok;
  TS_Pseudo      *pseudo = (TS_Pseudo*)ts->data;
  PetscReal      current_time_step;
  
  PetscFunctionBegin;
  *steps = -ts->steps;

  ierr = VecCopy(sol,pseudo->update);CHKERRQ(ierr);
  for (i=0; i<max_steps && ts->ptime < ts->max_time; i++) {
    ierr = TSPseudoComputeTimeStep(ts,&ts->time_step);CHKERRQ(ierr);
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
    current_time_step = ts->time_step;
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    while (PETSC_TRUE) {
      ts->ptime  += current_time_step;
      ierr = SNESSolve(ts->snes,PETSC_NULL,pseudo->update);CHKERRQ(ierr);
      ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
      ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
      ts->nonlinear_its += its; ts->linear_its += lits;
      ierr = TSPseudoVerifyTimeStep(ts,pseudo->update,&ts->time_step,&ok);CHKERRQ(ierr);
      if (ok) break;
      ts->ptime        -= current_time_step;
      current_time_step = ts->time_step;
    }
    ierr = VecCopy(pseudo->update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSPostStep(ts);CHKERRQ(ierr);
  }
  ierr = TSComputeRHSFunction(ts,ts->ptime,ts->vec_sol,pseudo->func);CHKERRQ(ierr);  
  ierr = VecNorm(pseudo->func,NORM_2,&pseudo->fnorm);CHKERRQ(ierr); 
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_Pseudo"
static PetscErrorCode TSDestroy_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pseudo->update) {ierr = VecDestroy(pseudo->update);CHKERRQ(ierr);}
  if (pseudo->func) {ierr = VecDestroy(pseudo->func);CHKERRQ(ierr);}
  if (pseudo->xdot) {ierr = VecDestroy(pseudo->xdot);CHKERRQ(ierr);}
  ierr = PetscFree(pseudo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoGetXdot"
/*
    Compute Xdot = (X^{n+1}-X^n)/dt) = 0
*/
static PetscErrorCode TSPseudoGetXdot(TS ts,Vec X,Vec *Xdot)
{
  TS_Pseudo      *pseudo = (TS_Pseudo*)ts->data;
  PetscScalar    mdt = 1.0/ts->time_step,*xnp1,*xn,*xdot;
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  ierr = VecGetArray(ts->vec_sol,&xn);CHKERRQ(ierr);
  ierr = VecGetArray(X,&xnp1);CHKERRQ(ierr);
  ierr = VecGetArray(pseudo->xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    xdot[i] = mdt*(xnp1[i] - xn[i]);
  }
  ierr = VecRestoreArray(ts->vec_sol,&xn);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&xnp1);CHKERRQ(ierr);
  ierr = VecRestoreArray(pseudo->xdot,&xdot);CHKERRQ(ierr);
  *Xdot = pseudo->xdot;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoFunction"
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
    algorithm, it only takes this one Newton step with the steady state
    residual, and then advances to the next time step.
*/
static PetscErrorCode TSPseudoFunction(SNES snes,Vec X,Vec Y,void *ctx)
{
  TS             ts = (TS)ctx;
  Vec            Xdot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPseudoGetXdot(ts,X,&Xdot);CHKERRQ(ierr);
  ierr = TSComputeIFunction(ts,ts->ptime,X,Xdot,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoJacobian"
/*
   This constructs the Jacobian needed for SNES.  For DAE, this is

       dF(X,Xdot)/dX + shift*dF(X,Xdot)/dXdot

    and for ODE:

       J = I/dt - J_{Frhs}   where J_{Frhs} is the given Jacobian of Frhs.
*/
static PetscErrorCode TSPseudoJacobian(SNES snes,Vec X,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS             ts = (TS)ctx;
  Vec            Xdot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPseudoGetXdot(ts,X,&Xdot);CHKERRQ(ierr);
  ierr = TSComputeIJacobian(ts,ts->ptime,X,Xdot,1./ts->time_step,AA,BB,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_Pseudo"
static PetscErrorCode TSSetUp_Pseudo(TS ts)
{
  TS_Pseudo      *pseudo = (TS_Pseudo*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&pseudo->update);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&pseudo->func);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&pseudo->xdot);CHKERRQ(ierr);
  ierr = SNESSetFunction(ts->snes,pseudo->func,TSPseudoFunction,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(ts->snes,ts->Arhs,ts->B,TSPseudoJacobian,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoMonitorDefault"
PetscErrorCode TSPseudoMonitorDefault(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  TS_Pseudo               *pseudo = (TS_Pseudo*)ts->data;
  PetscErrorCode          ierr;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor)ctx;

  PetscFunctionBegin;
  if (!ctx) {
    ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ts)->comm,"stdout",0,&viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIMonitorPrintf(viewer,"TS %D dt %G time %G fnorm %G\n",step,ts->time_step,ptime,pseudo->fnorm);CHKERRQ(ierr);
  if (!ctx) {
    ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_Pseudo"
static PetscErrorCode TSSetFromOptions_Pseudo(TS ts)
{
  TS_Pseudo               *pseudo = (TS_Pseudo*)ts->data;
  PetscErrorCode          ierr;
  PetscTruth              flg = PETSC_FALSE;
  PetscViewerASCIIMonitor viewer;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Pseudo-timestepping options");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ts_monitor","Monitor convergence","TSPseudoMonitorDefault",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ts)->comm,"stdout",0,&viewer);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSPseudoMonitorDefault,viewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ts_pseudo_increment_dt_from_initial_dt","Increase dt as a ratio from original dt","TSPseudoIncrementDtFromInitialDt",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = TSPseudoIncrementDtFromInitialDt(ts);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-ts_pseudo_increment","Ratio to increase dt","TSPseudoSetTimeStepIncrement",pseudo->dt_increment,&pseudo->dt_increment,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_Pseudo"
static PetscErrorCode TSView_Pseudo(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoSetVerifyTimeStep"
/*@C
   TSPseudoSetVerifyTimeStep - Sets a user-defined routine to verify the quality of the 
   last timestep.

   Collective on TS

   Input Parameters:
+  ts - timestep context
.  dt - user-defined function to verify timestep
-  ctx - [optional] user-defined context for private data
         for the timestep verification routine (may be PETSC_NULL)

   Level: advanced

   Calling sequence of func:
.  func (TS ts,Vec update,void *ctx,PetscReal *newdt,PetscTruth *flag);

.  update - latest solution vector
.  ctx - [optional] timestep context
.  newdt - the timestep to use for the next step
.  flag - flag indicating whether the last time step was acceptable

   Notes:
   The routine set here will be called by TSPseudoVerifyTimeStep()
   during the timestepping process.

.keywords: timestep, pseudo, set, verify 

.seealso: TSPseudoDefaultVerifyTimeStep(), TSPseudoVerifyTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoSetVerifyTimeStep(TS ts,PetscErrorCode (*dt)(TS,Vec,void*,PetscReal*,PetscTruth*),void* ctx)
{
  PetscErrorCode ierr,(*f)(TS,PetscErrorCode (*)(TS,Vec,void*,PetscReal *,PetscTruth *),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoSetVerifyTimeStep_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,dt,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoSetTimeStepIncrement"
/*@
    TSPseudoSetTimeStepIncrement - Sets the scaling increment applied to 
    dt when using the TSPseudoDefaultTimeStep() routine.

   Collective on TS

    Input Parameters:
+   ts - the timestep context
-   inc - the scaling factor >= 1.0

    Options Database Key:
$    -ts_pseudo_increment <increment>

    Level: advanced

.keywords: timestep, pseudo, set, increment

.seealso: TSPseudoSetTimeStep(), TSPseudoDefaultTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoSetTimeStepIncrement(TS ts,PetscReal inc)
{
  PetscErrorCode ierr,(*f)(TS,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoSetTimeStepIncrement_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,inc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoIncrementDtFromInitialDt"
/*@
    TSPseudoIncrementDtFromInitialDt - Indicates that a new timestep
    is computed via the formula
$         dt = initial_dt*initial_fnorm/current_fnorm 
      rather than the default update,
$         dt = current_dt*previous_fnorm/current_fnorm.

   Collective on TS

    Input Parameter:
.   ts - the timestep context

    Options Database Key:
$    -ts_pseudo_increment_dt_from_initial_dt

    Level: advanced

.keywords: timestep, pseudo, set, increment

.seealso: TSPseudoSetTimeStep(), TSPseudoDefaultTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoIncrementDtFromInitialDt(TS ts)
{
  PetscErrorCode ierr,(*f)(TS);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoIncrementDtFromInitialDt_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSPseudoSetTimeStep"
/*@C
   TSPseudoSetTimeStep - Sets the user-defined routine to be
   called at each pseudo-timestep to update the timestep.

   Collective on TS

   Input Parameters:
+  ts - timestep context
.  dt - function to compute timestep
-  ctx - [optional] user-defined context for private data
         required by the function (may be PETSC_NULL)

   Level: intermediate

   Calling sequence of func:
.  func (TS ts,PetscReal *newdt,void *ctx);

.  newdt - the newly computed timestep
.  ctx - [optional] timestep context

   Notes:
   The routine set here will be called by TSPseudoComputeTimeStep()
   during the timestepping process.

.keywords: timestep, pseudo, set

.seealso: TSPseudoDefaultTimeStep(), TSPseudoComputeTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoSetTimeStep(TS ts,PetscErrorCode (*dt)(TS,PetscReal*,void*),void* ctx)
{
  PetscErrorCode ierr,(*f)(TS,PetscErrorCode (*)(TS,PetscReal *,void *),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE,1);

  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSPseudoSetTimeStep_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ts,dt,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------- */

typedef PetscErrorCode (*FCN1)(TS,Vec,void*,PetscReal*,PetscTruth*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoSetVerifyTimeStep_Pseudo"
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoSetVerifyTimeStep_Pseudo(TS ts,FCN1 dt,void* ctx)
{
  TS_Pseudo *pseudo;

  PetscFunctionBegin;
  pseudo              = (TS_Pseudo*)ts->data;
  pseudo->verify      = dt;
  pseudo->verifyctx   = ctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoSetTimeStepIncrement_Pseudo"
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoSetTimeStepIncrement_Pseudo(TS ts,PetscReal inc)
{
  TS_Pseudo *pseudo = (TS_Pseudo*)ts->data;

  PetscFunctionBegin;
  pseudo->dt_increment = inc;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoIncrementDtFromInitialDt_Pseudo"
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoIncrementDtFromInitialDt_Pseudo(TS ts)
{
  TS_Pseudo *pseudo = (TS_Pseudo*)ts->data;

  PetscFunctionBegin;
  pseudo->increment_dt_from_initial_dt = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN2)(TS,PetscReal*,void*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoSetTimeStep_Pseudo"
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoSetTimeStep_Pseudo(TS ts,FCN2 dt,void* ctx)
{
  TS_Pseudo *pseudo = (TS_Pseudo*)ts->data;

  PetscFunctionBegin;
  pseudo->dt      = dt;
  pseudo->dtctx   = ctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ----------------------------------------------------------------------------- */
/*MC
      TSPSEUDO - Solve steady state ODE and DAE problems with pseudo time stepping

  This method solves equations of the form

$    F(X,Xdot) = 0

  for steady state using the iteration

$    [G'] S = -F(X,0)
$    X += S

  where

$    G(Y) = F(Y,(Y-X)/dt)

  This is linearly-implicit Euler with the residual always evaluated "at steady
  state".  See note below.

  Options database keys:
+  -ts_pseudo_increment <real> - ratio of increase dt
-  -ts_pseudo_increment_dt_from_initial_dt <truth> - Increase dt as a ratio from original dt

  Level: beginner

  References:
  Todd S. Coffey and C. T. Kelley and David E. Keyes, Pseudotransient Continuation and Differential-Algebraic Equations, 2003.
  C. T. Kelley and David E. Keyes, Convergence analysis of Pseudotransient Continuation, 1998.

  Notes:
  The residual computed by this method includes the transient term (Xdot is computed instead of
  always being zero), but since the prediction from the last step is always the solution from the
  last step, on the first Newton iteration we have

$  Xdot = (Xpredicted - Xold)/dt = (Xold-Xold)/dt = 0

  Therefore, the linear system solved by the first Newton iteration is equivalent to the one
  described above and in the papers.  If the user chooses to perform multiple Newton iterations, the
  algorithm is no longer the one described in the referenced papers.

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_Pseudo"
PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Pseudo(TS ts)
{
  TS_Pseudo      *pseudo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->destroy         = TSDestroy_Pseudo;
  ts->ops->view            = TSView_Pseudo;

  if (ts->problem_type == TS_LINEAR) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Only for nonlinear problems");
  }
  if (!ts->Arhs) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set Jacobian");
  }

  ts->ops->setup           = TSSetUp_Pseudo;  
  ts->ops->step            = TSStep_Pseudo;
  ts->ops->setfromoptions  = TSSetFromOptions_Pseudo;

  /* create the required nonlinear solver context */
  ierr = SNESCreate(((PetscObject)ts)->comm,&ts->snes);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ts->snes,(PetscObject)ts,1);CHKERRQ(ierr);

  ierr = PetscNewLog(ts,TS_Pseudo,&pseudo);CHKERRQ(ierr);
  ts->data = (void*)pseudo;

  pseudo->dt_increment                 = 1.1;
  pseudo->increment_dt_from_initial_dt = PETSC_FALSE;
  pseudo->dt                           = TSPseudoDefaultTimeStep;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPseudoSetVerifyTimeStep_C",
                    "TSPseudoSetVerifyTimeStep_Pseudo",
                     TSPseudoSetVerifyTimeStep_Pseudo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPseudoSetTimeStepIncrement_C",
                    "TSPseudoSetTimeStepIncrement_Pseudo",
                     TSPseudoSetTimeStepIncrement_Pseudo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPseudoIncrementDtFromInitialDt_C",
                    "TSPseudoIncrementDtFromInitialDt_Pseudo",
                     TSPseudoIncrementDtFromInitialDt_Pseudo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSPseudoSetTimeStep_C",
                    "TSPseudoSetTimeStep_Pseudo",
                     TSPseudoSetTimeStep_Pseudo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "TSPseudoDefaultTimeStep"
/*@C
   TSPseudoDefaultTimeStep - Default code to compute pseudo-timestepping.
   Use with TSPseudoSetTimeStep().

   Collective on TS

   Input Parameters:
.  ts - the timestep context
.  dtctx - unused timestep context

   Output Parameter:
.  newdt - the timestep to use for the next step

   Level: advanced

.keywords: timestep, pseudo, default

.seealso: TSPseudoSetTimeStep(), TSPseudoComputeTimeStep()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSPseudoDefaultTimeStep(TS ts,PetscReal* newdt,void* dtctx)
{
  TS_Pseudo      *pseudo = (TS_Pseudo*)ts->data;
  PetscReal      inc = pseudo->dt_increment,fnorm_previous = pseudo->fnorm_previous;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSComputeRHSFunction(ts,ts->ptime,ts->vec_sol,pseudo->func);CHKERRQ(ierr);  
  ierr = VecNorm(pseudo->func,NORM_2,&pseudo->fnorm);CHKERRQ(ierr); 
  if (pseudo->initial_fnorm == 0.0) {
    /* first time through so compute initial function norm */
    pseudo->initial_fnorm = pseudo->fnorm;
    fnorm_previous        = pseudo->fnorm;
  }
  if (pseudo->fnorm == 0.0) {
    *newdt = 1.e12*inc*ts->time_step; 
  } else if (pseudo->increment_dt_from_initial_dt) {
    *newdt = inc*ts->initial_time_step*pseudo->initial_fnorm/pseudo->fnorm;
  } else {
    *newdt = inc*ts->time_step*fnorm_previous/pseudo->fnorm;
  }
  pseudo->fnorm_previous = pseudo->fnorm;
  PetscFunctionReturn(0);
}
