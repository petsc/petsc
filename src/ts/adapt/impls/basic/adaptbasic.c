#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/
#include <petscdm.h>

static PetscErrorCode TSAdaptChoose_Basic(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  Vec            Y;
  DM             dm;
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
    if (enorm >= 0 && order < 1) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed error order %D must be positive",order);
  } else if (ts->ops->evaluatestep) {
    if (adapt->candidates.n < 1) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"No candidate has been registered");
    if (!adapt->candidates.inuse_set) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
    order = adapt->candidates.order[0];
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&Y);CHKERRQ(ierr);
    ierr = TSEvaluateStep(ts,order-1,Y,NULL);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,ts->vec_sol,Y,adapt->wnormtype,&enorm,&enorma,&enormr);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&Y);CHKERRQ(ierr);
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
      ierr = PetscInfo(adapt,"Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (adapt->always_accept) {
      ierr = PetscInfo(adapt,"Estimated scaled local truncation error %g, accepting step of size %g because always_accept is set\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else {
      ierr = PetscInfo(adapt,"Estimated scaled local truncation error %g, rejecting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
    }
  } else {
    ierr = PetscInfo(adapt,"Estimated scaled local truncation error %g, accepting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
    *accept = PETSC_TRUE;
  }

  /* The optimal new step based purely on local truncation error for this step. */
  if (enorm > 0)
    hfac_lte = safety * PetscPowReal(enorm,((PetscReal)-1)/order);
  else
    hfac_lte = safety * PETSC_INFINITY;
  if (adapt->timestepjustdecreased) {
    hfac_lte = PetscMin(hfac_lte,1.0);
    adapt->timestepjustdecreased--;
  }
  h_lte = h * PetscClipInterval(hfac_lte,adapt->clip[0],adapt->clip[1]);

  *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  *wlte   = enorm;
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTBASIC - Basic adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSGetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Basic(TSAdapt adapt)
{
  PetscFunctionBegin;
  adapt->ops->choose = TSAdaptChoose_Basic;
  PetscFunctionReturn(0);
}
