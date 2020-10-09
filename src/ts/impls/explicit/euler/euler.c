/*
       Code for Timestepping with explicit Euler.
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef struct {
  Vec update;     /* work vector where new solution is formed  */
} TS_Euler;

static PetscErrorCode TSStep_Euler(TS ts)
{
  TS_Euler       *euler = (TS_Euler*)ts->data;
  Vec            solution = ts->vec_sol,update = euler->update;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPreStage(ts,ts->ptime);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts,ts->ptime,solution,update);CHKERRQ(ierr);
  ierr = VecAYPX(update,ts->time_step,solution);CHKERRQ(ierr);
  ierr = TSPostStage(ts,ts->ptime,0,&solution);CHKERRQ(ierr);
  ierr = TSAdaptCheckStage(ts->adapt,ts,ts->ptime,solution,&stageok);CHKERRQ(ierr);
  if (!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}
  ierr = TSFunctionDomainError(ts,ts->ptime+ts->time_step,update,&stageok);CHKERRQ(ierr);
  if (!stageok) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}

  ierr = TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
  if (!accept) {ts->reason = TS_DIVERGED_STEP_REJECTED; PetscFunctionReturn(0);}
  ierr = VecCopy(update,solution);CHKERRQ(ierr);

  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetUp_Euler(TS ts)
{
  TS_Euler       *euler = (TS_Euler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCheckImplicitTerm(ts);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&euler->update);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_Euler(TS ts)
{
  TS_Euler       *euler = (TS_Euler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&euler->update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Euler(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Euler(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetFromOptions_Euler(PetscOptionItems *PetscOptionsObject,TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Euler(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_Euler(TS ts,PetscReal t,Vec X)
{
  TS_Euler       *euler = (TS_Euler*)ts->data;
  Vec            update = euler->update;
  PetscReal      alpha = (ts->ptime - t)/ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecWAXPY(X,-ts->time_step,update,ts->vec_sol);CHKERRQ(ierr);
  ierr = VecAXPBY(X,1.0-alpha,alpha,ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeLinearStability_Euler(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

/*MC
      TSEULER - ODE solver using the explicit forward Euler method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSBEULER

M*/
PETSC_EXTERN PetscErrorCode TSCreate_Euler(TS ts)
{
  TS_Euler       *euler;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(ts,&euler);CHKERRQ(ierr);
  ts->data = (void*)euler;

  ts->ops->setup           = TSSetUp_Euler;
  ts->ops->step            = TSStep_Euler;
  ts->ops->reset           = TSReset_Euler;
  ts->ops->destroy         = TSDestroy_Euler;
  ts->ops->setfromoptions  = TSSetFromOptions_Euler;
  ts->ops->view            = TSView_Euler;
  ts->ops->interpolate     = TSInterpolate_Euler;
  ts->ops->linearstability = TSComputeLinearStability_Euler;
  ts->default_adapt_type   = TSADAPTNONE;
  ts->usessnes             = PETSC_FALSE;
  PetscFunctionReturn(0);
}
