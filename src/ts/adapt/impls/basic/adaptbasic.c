#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  Vec Y;
} TSAdapt_Basic;

static PetscErrorCode TSAdaptChoose_Basic(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscInt       order  = PETSC_DECIDE;
  PetscReal      enorm  = -1;
  PetscReal      enorma,enormr;
  PetscReal      safety = adapt->safety;
  PetscReal      hfac_lte,h_lte;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *next_sc = 0;   /* Reuse the same order scheme */
  *wltea   = -1;  /* Weighted absolute local truncation error is not used */
  *wlter   = -1;  /* Weighted relative local truncation error is not used */

  if (ts->ops->evaluatewlte) {
    ierr = TSEvaluateWLTE(ts,adapt->wnormtype,&order,&enorm);CHKERRQ(ierr);
    if (enorm >= 0 && order < 1) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed error order %D must be positive",order);
  } else if (ts->ops->evaluatestep) {
    if (adapt->candidates.n < 1) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"No candidate has been registered");
    if (!adapt->candidates.inuse_set) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
    if (!basic->Y) {ierr = VecDuplicate(ts->vec_sol,&basic->Y);CHKERRQ(ierr);}
    order = adapt->candidates.order[0];
    ierr = TSEvaluateStep(ts,order-1,basic->Y,NULL);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,ts->vec_sol,basic->Y,adapt->wnormtype,&enorm,&enorma,&enormr);CHKERRQ(ierr);
  }

  if (enorm < 0) {
    *accept  = PETSC_TRUE;
    *next_h  = h;            /* Reuse the old step */
    *wlte    = -1;           /* Weighted local truncation error was not evaluated */
    PetscFunctionReturn(0);
  }

  /* Determine whether the step is accepted of rejected */
  if (enorm > 1) {
    if (!*accept) safety *= adapt->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (adapt->always_accept) {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g because always_accept is set\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, rejecting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
    }
  } else {
    ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
    *accept = PETSC_TRUE;
  }

  /* The optimal new step based purely on local truncation error for this step. */
  if (enorm > 0)
    hfac_lte = safety * PetscPowReal(enorm,((PetscReal)-1)/order);
  else
    hfac_lte = safety * PETSC_INFINITY;
  if (adapt->timestepjustincreased){
    hfac_lte = PetscMin(hfac_lte,1.0);
    adapt->timestepjustincreased--;
  }
  h_lte = h * PetscClipInterval(hfac_lte,adapt->clip[0],adapt->clip[1]);

  *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  *wlte   = enorm;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptReset_Basic(TSAdapt adapt)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&basic->Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDestroy_Basic(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptReset_Basic(adapt);CHKERRQ(ierr);
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTBASIC - Basic adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Basic(TSAdapt adapt)
{
  PetscErrorCode ierr;
  TSAdapt_Basic  *basic;

  PetscFunctionBegin;
  ierr= PetscNewLog(adapt,&basic);CHKERRQ(ierr);
  adapt->data         = (void*)basic;
  adapt->ops->choose  = TSAdaptChoose_Basic;
  adapt->ops->reset   = TSAdaptReset_Basic;
  adapt->ops->destroy = TSAdaptDestroy_Basic;
  PetscFunctionReturn(0);
}
