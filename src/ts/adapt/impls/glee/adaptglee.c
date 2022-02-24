#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/
#include <petscdm.h>

typedef struct {
  Vec Y;
} TSAdapt_GLEE;

static PetscErrorCode TSAdaptChoose_GLEE(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  TSAdapt_GLEE   *glee = (TSAdapt_GLEE*)adapt->data;
  Vec            X,Y,E;
  PetscReal      enorm,enorma,enormr,hfac_lte,hfac_ltea,hfac_lter,h_lte,safety;
  PetscInt       order;
  PetscBool      bGTEMethod;

  PetscFunctionBegin;
  *next_sc = 0; /* Reuse the same order scheme */
  safety = adapt->safety;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)ts,TSGLEE,&bGTEMethod));
  order = adapt->candidates.order[0];

  if (bGTEMethod) {/* the method is of GLEE type */
    DM dm;

    CHKERRQ(TSGetSolution(ts,&X));
    if (!glee->Y && adapt->glee_use_local) {
      CHKERRQ(VecDuplicate(X,&glee->Y));/*create vector to store previous step global error*/
      CHKERRQ(VecZeroEntries(glee->Y)); /*set error to zero on the first step - may not work if error is not zero initially*/
    }
    CHKERRQ(TSGetDM(ts,&dm));
    CHKERRQ(DMGetGlobalVector(dm,&E));
    CHKERRQ(TSGetTimeError(ts,0,&E));

    if (adapt->glee_use_local) CHKERRQ(VecAXPY(E,-1.0,glee->Y)); /* local error = current error - previous step error */

    /* this should be called with the solution at the beginning of the step too*/
    CHKERRQ(TSErrorWeightedENorm(ts,E,X,X,adapt->wnormtype,&enorm,&enorma,&enormr));
    CHKERRQ(DMRestoreGlobalVector(dm,&E));
  } else {
    /* the method is NOT of GLEE type; use the stantard basic augmented by separate atol and rtol */
    CHKERRQ(TSGetSolution(ts,&X));
    if (!glee->Y) CHKERRQ(VecDuplicate(X,&glee->Y));
    Y     = glee->Y;
    CHKERRQ(TSEvaluateStep(ts,order-1,Y,NULL));
    CHKERRQ(TSErrorWeightedNorm(ts,X,Y,adapt->wnormtype,&enorm,&enorma,&enormr));
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
      CHKERRQ(PetscInfo(adapt,"Estimated scaled truncation error [combined, absolute, relative]] [%g, %g, %g], accepting because step size %g is at minimum\n",(double)enorm,(double)enorma,(double)enormr,(double)h));
      *accept = PETSC_TRUE;
    } else if (adapt->always_accept) {
      CHKERRQ(PetscInfo(adapt,"Estimated scaled truncation error [combined, absolute, relative]] [%g, %g, %g], accepting step of size %g because always_accept is set\n",(double)enorm,(double)enorma,(double)enormr,(double)h));
      *accept = PETSC_TRUE;
    } else {
      CHKERRQ(PetscInfo(adapt,"Estimated scaled truncation error [combined, absolute, relative]] [%g, %g, %g], rejecting step of size %g\n",(double)enorm,(double)enorma,(double)enormr,(double)h));
      *accept = PETSC_FALSE;
    }
  } else {
    CHKERRQ(PetscInfo(adapt,"Estimated scaled truncation error [combined, absolute, relative] [%g, %g, %g], accepting step of size %g\n",(double)enorm,(double)enorma,(double)enormr,(double)h));
    *accept = PETSC_TRUE;
  }

  if (bGTEMethod) {
    if (*accept == PETSC_TRUE && adapt->glee_use_local) {
      /* If step is accepted, then overwrite previous step error with the current error to be used on the next step */
      /* WARNING: if the adapters are composable, then the accept test will not be reliable*/
      CHKERRQ(TSGetTimeError(ts,0,&glee->Y));
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

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&glee->Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDestroy_GLEE(TSAdapt adapt)
{
  PetscFunctionBegin;
  CHKERRQ(TSAdaptReset_GLEE(adapt));
  CHKERRQ(PetscFree(adapt->data));
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTGLEE - GLEE adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSGetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_GLEE(TSAdapt adapt)
{
  TSAdapt_GLEE  *glee;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(adapt,&glee));
  adapt->data         = (void*)glee;
  adapt->ops->choose  = TSAdaptChoose_GLEE;
  adapt->ops->reset   = TSAdaptReset_GLEE;
  adapt->ops->destroy = TSAdaptDestroy_GLEE;
  PetscFunctionReturn(0);
}
