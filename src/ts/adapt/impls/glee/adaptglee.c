#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  Vec E,Y;
} TSAdapt_GLEE;

static PetscErrorCode TSAdaptChoose_GLEE(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  TSAdapt_GLEE  *glee = (TSAdapt_GLEE*)adapt->data;
  TSType         time_scheme;      /* Type of time-integration scheme        */
  PetscErrorCode ierr;
  Vec            X,Y,E;
  PetscReal      enorm,enorma,enormr,hfac_lte,hfac_ltea,hfac_lter,h_lte,safety;
  PetscInt       order;
  PetscBool      bGTEMethod=PETSC_FALSE;

  PetscFunctionBegin;

  *next_sc = 0; /* Reuse the same order scheme */
  safety = adapt->safety;
  ierr = TSGetType(ts,&time_scheme);CHKERRQ(ierr);
  if (!strcmp(time_scheme,TSGLEE)) bGTEMethod=PETSC_TRUE;
  order = adapt->candidates.order[0];

  if (bGTEMethod){/* the method is of GLEE type */
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!glee->Y && adapt->glee_use_local) {
      ierr = VecDuplicate(X,&glee->Y);CHKERRQ(ierr);/*create vector to store previous step global error*/
      ierr = VecZeroEntries(glee->Y);CHKERRQ(ierr); /*set error to zero on the first step - may not work if error is not zero initially*/
    }
    if (!glee->E) {ierr = VecDuplicate(X,&glee->E);CHKERRQ(ierr);}
    E    = glee->E;
    ierr = TSGetTimeError(ts,0,&E);CHKERRQ(ierr);

    if (adapt->glee_use_local) {ierr = VecAXPY(E,-1.0,glee->Y);CHKERRQ(ierr);} /* local error = current error - previous step error */

    /* this should be called with the solution at the beginning of the step too*/
    ierr = TSErrorWeightedENorm(ts,E,X,X,adapt->wnormtype,&enorm,&enorma,&enormr);CHKERRQ(ierr);
  } else {
    /* the method is NOT of GLEE type; use the stantard basic augmented by separate atol and rtol */
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!glee->Y) {ierr = VecDuplicate(X,&glee->Y);CHKERRQ(ierr);}
    Y     = glee->Y;
    ierr  = TSEvaluateStep(ts,order-1,Y,NULL);CHKERRQ(ierr);
    ierr  = TSErrorWeightedNorm(ts,X,Y,adapt->wnormtype,&enorm,&enorma,&enormr);CHKERRQ(ierr);
  }

  if (enorm < 0) {
    *accept  = PETSC_TRUE;
    *next_h  = h;            /* Reuse the old step */
    *wlte    = -1;           /* Weighted error was not evaluated */
    *wltea   = -1;           /* Weighted absolute error was not evaluated */
    *wlter   = -1;           /* Weighted relative error was not evaluated */
    PetscFunctionReturn(0);
  }

  if (enorm > 1. || enorma > 1. || enormr > 1.) {
    if (!*accept) safety *= adapt->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr    = PetscInfo4(adapt,"Estimated scaled truncation error [combined, absolute, relative]] [%g, %g, %g], accepting because step size %g is at minimum\n",(double)enorm,(double)enorma,(double)enormr,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (adapt->always_accept) {
      ierr    = PetscInfo4(adapt,"Estimated scaled truncation error [combined, absolute, relative]] [%g, %g, %g], accepting step of size %g because always_accept is set\n",(double)enorm,(double)enorma,(double)enormr,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else {
      ierr    = PetscInfo4(adapt,"Estimated scaled truncation error [combined, absolute, relative]] [%g, %g, %g], rejecting step of size %g\n",(double)enorm,(double)enorma,(double)enormr,(double)h);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
    }
  } else {
    ierr    = PetscInfo4(adapt,"Estimated scaled truncation error [combined, absolute, relative] [%g, %g, %g], accepting step of size %g\n",(double)enorm,(double)enorma,(double)enormr,(double)h);CHKERRQ(ierr);
    *accept = PETSC_TRUE;
  }

  if (bGTEMethod){
    if (*accept == PETSC_TRUE && adapt->glee_use_local) {
      /* If step is accepted, then overwrite previous step error with the current error to be used on the next step */
      /* WARNING: if the adapters are composable, then the accept test will not be reliable*/
      ierr = TSGetTimeError(ts,0,&(glee->Y));CHKERRQ(ierr);
    }

    /* The optimal new step based on the current global truncation error. */
    if (enorm > 0) {
      /* factor based on the absolute tolerance */
      hfac_ltea = safety * PetscPowReal(1./enorma,((PetscReal)1)/(order+1));
      /* factor based on the relative tolerance */
      hfac_lter = safety * PetscPowReal(1./enormr,((PetscReal)1)/(order+1));
      /* pick the minimum time step among the relative and absolute tolerances */
      hfac_lte  = PetscMin(hfac_ltea,hfac_lter);
    } else {
      hfac_lte = safety * PETSC_INFINITY;
    }
    h_lte = h * PetscClipInterval(hfac_lte,adapt->clip[0],adapt->clip[1]);
    *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  } else {
    /* The optimal new step based purely on local truncation error for this step. */
    if (enorm > 0) {
      /* factor based on the absolute tolerance */
      hfac_ltea = safety * PetscPowReal(enorma,((PetscReal)-1)/order);
      /* factor based on the relative tolerance */
      hfac_lter = safety * PetscPowReal(enormr,((PetscReal)-1)/order);
      /* pick the minimum time step among the relative and absolute tolerances */
      hfac_lte  = PetscMin(hfac_ltea,hfac_lter);
    } else {
      hfac_lte = safety * PETSC_INFINITY;
    }
    h_lte = h * PetscClipInterval(hfac_lte,adapt->clip[0],adapt->clip[1]);
    *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  }
  *wlte   = enorm;
  *wltea  = enorma;
  *wlter  = enormr;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptReset_GLEE(TSAdapt adapt)
{
  TSAdapt_GLEE  *glee = (TSAdapt_GLEE*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&glee->Y);CHKERRQ(ierr);
  ierr = VecDestroy(&glee->E);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDestroy_GLEE(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptReset_GLEE(adapt);CHKERRQ(ierr);
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTGLEE - GLEE adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSGetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_GLEE(TSAdapt adapt)
{
  PetscErrorCode ierr;
  TSAdapt_GLEE  *glee;

  PetscFunctionBegin;
  ierr = PetscNewLog(adapt,&glee);CHKERRQ(ierr);
  adapt->data         = (void*)glee;
  adapt->ops->choose  = TSAdaptChoose_GLEE;
  adapt->ops->reset   = TSAdaptReset_GLEE;
  adapt->ops->destroy = TSAdaptDestroy_GLEE;
  PetscFunctionReturn(0);
}
