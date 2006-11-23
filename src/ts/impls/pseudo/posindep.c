#define PETSCTS_DLL

/*
       Code for Timestepping with implicit backwards Euler.
*/
#include "include/private/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */

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
    while (PETSC_TRUE) {
      ts->ptime  += current_time_step;
      ierr = SNESSolve(ts->snes,PETSC_NULL,pseudo->update);CHKERRQ(ierr);
      ierr = SNESGetNumberLinearIterations(ts->snes,&lits);CHKERRQ(ierr);
      ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
      ts->nonlinear_its += its; ts->linear_its += lits;
      ierr = TSPseudoVerifyTimeStep(ts,pseudo->update,&ts->time_step,&ok);CHKERRQ(ierr);
      if (ok) break;
      ts->ptime        -= current_time_step;
      current_time_step = ts->time_step;
    }
    ierr = VecCopy(pseudo->update,sol);CHKERRQ(ierr);
    ts->steps++;
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
  if (pseudo->rhs)  {ierr = VecDestroy(pseudo->rhs);CHKERRQ(ierr);}
  ierr = PetscFree(pseudo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/

/* 
    This defines the nonlinear equation that is to be solved with SNES

              (U^{n+1} - U^{n})/dt - F(U^{n+1})
*/
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoFunction"
PetscErrorCode TSPseudoFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS             ts = (TS) ctx;
  PetscScalar    mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  PetscErrorCode ierr;
  PetscInt       i,n;

  PetscFunctionBegin;
  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y);CHKERRQ(ierr);
  /* compute (u^{n+1) - u^{n})/dt - F(u^{n+1}) */
  ierr = VecGetArray(ts->vec_sol,&un);CHKERRQ(ierr);
  ierr = VecGetArray(x,&unp1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&Funp1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    Funp1[i] = mdt*(unp1[i] - un[i]) - Funp1[i];
  }
  ierr = VecRestoreArray(ts->vec_sol,&un);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&unp1);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&Funp1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   This constructs the Jacobian needed for SNES 

             J = I/dt - J_{F}   where J_{F} is the given Jacobian of F.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSPseudoJacobian"
PetscErrorCode TSPseudoJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS             ts = (TS) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* construct users Jacobian */
  ierr = TSComputeRHSJacobian(ts,ts->ptime,x,AA,BB,str);CHKERRQ(ierr);

  /* shift and scale Jacobian */
  ierr = TSScaleShiftMatrices(ts,*AA,*BB,*str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_Pseudo"
static PetscErrorCode TSSetUp_Pseudo(TS ts)
{
  TS_Pseudo      *pseudo = (TS_Pseudo*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr); */
  ierr = VecDuplicate(ts->vec_sol,&pseudo->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&pseudo->func);CHKERRQ(ierr);  
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
    ierr = PetscViewerASCIIMonitorCreate(ts->comm,"stdout",0,&viewer);CHKERRQ(ierr);
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
  PetscTruth              flg;
  PetscViewerASCIIMonitor viewer;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Pseudo-timestepping options");CHKERRQ(ierr);
    ierr = PetscOptionsName("-ts_monitor","Monitor convergence","TSPseudoMonitorDefault",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(ts->comm,"stdout",0,&viewer);CHKERRQ(ierr);
      ierr = TSMonitorSet(ts,TSPseudoMonitorDefault,viewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-ts_pseudo_increment_dt_from_initial_dt","Increase dt as a ratio from original dt","TSPseudoIncrementDtFromInitialDt",&flg);CHKERRQ(ierr);
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
  ierr = SNESCreate(ts->comm,&ts->snes);CHKERRQ(ierr);

  ierr = PetscNew(TS_Pseudo,&pseudo);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ts,sizeof(TS_Pseudo));CHKERRQ(ierr);

  ierr     = PetscMemzero(pseudo,sizeof(TS_Pseudo));CHKERRQ(ierr);
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

















