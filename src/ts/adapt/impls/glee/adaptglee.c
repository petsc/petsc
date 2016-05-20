#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  PetscBool always_accept;
  PetscReal clip[2];            /* admissible decrease/increase factors */
  PetscReal safety;             /* safety factor relative to target error */
  PetscReal reject_safety;      /* extra safety factor if the last step was rejected */
  Vec       X;
  Vec       Y;
  Vec       E;
} TSAdapt_GLEE;

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_GLEE"
static PetscErrorCode TSAdaptChoose_GLEE(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  TSAdapt_GLEE  *glee = (TSAdapt_GLEE*)adapt->data;
  TSType         time_scheme;      /* Type of time-integration scheme        */
  PetscErrorCode ierr;
  Vec            X,Y,E;
  PetscReal      enorm,enorma,enormr,hfac_lte,hfac_ltea,hfac_lter,h_lte,safety;
  PetscInt       order,stepno;
  PetscBool      bGTEMethod=PETSC_FALSE;

  PetscFunctionBegin;

  *next_sc = 0; /* Reuse the same order scheme */
  safety = glee->safety;
  ierr = TSGetTimeStepNumber(ts,&stepno);CHKERRQ(ierr);
  ierr = TSGetType(ts,&time_scheme);CHKERRQ(ierr);
  if (!strcmp(time_scheme,TSGLEE)) bGTEMethod=PETSC_TRUE;
  order = adapt->candidates.order[0];

  if (bGTEMethod){/* the method is of GLEE type */
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!glee->E) {ierr = VecDuplicate(X,&glee->E);CHKERRQ(ierr);}
    E     = glee->E;
    ierr = TSGetTimeError(ts,0,&E);CHKERRQ(ierr);
    /* this should be called with Y (the solution at the beginning of the step)*/
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
    if (!*accept) safety *= glee->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr    = PetscInfo4(adapt,"Estimated scaled truncation error [combined, absolute, relative]] [%g, %g, %g], accepting because step size %g is at minimum\n",(double)enorm,(double)enorma,(double)enormr,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (glee->always_accept) {
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
    /* The optimal new step based on the current global truncation error. */
    if (enorm > 0) {
      /* factor based on the absolute tolerance */
      hfac_ltea = safety * PetscPowReal(1./enorma,((PetscReal)1)/order);
      /* factor based on the relative tolerance */
      hfac_lter = safety * PetscPowReal(1./enormr,((PetscReal)1)/order);
      /* pick the minimum time step among the relative and absolute tolerances */
      hfac_lte  = PetscMin(hfac_ltea,hfac_lter);
    } else {
      hfac_lte = safety * PETSC_INFINITY;
    }
    h_lte = h * PetscClipInterval(hfac_lte,glee->clip[0],glee->clip[1]);
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
    h_lte = h * PetscClipInterval(hfac_lte,glee->clip[0],glee->clip[1]);
    *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  }
  *wlte   = enorm;
  *wltea  = enorma;
  *wlter  = enormr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptReset_GLEE"
static PetscErrorCode TSAdaptReset_GLEE(TSAdapt adapt)
{
  TSAdapt_GLEE  *glee = (TSAdapt_GLEE*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&glee->Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptDestroy_GLEE"
static PetscErrorCode TSAdaptDestroy_GLEE(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptReset_GLEE(adapt);CHKERRQ(ierr);
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetFromOptions_GLEE"
static PetscErrorCode TSAdaptSetFromOptions_GLEE(PetscOptionItems *PetscOptionsObject,TSAdapt adapt)
{
  TSAdapt_GLEE  *glee = (TSAdapt_GLEE*)adapt->data;
  PetscErrorCode ierr;
  PetscInt       two;
  PetscBool      set;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"GLEE adaptive controller options");CHKERRQ(ierr);
  two  = 2;
  ierr = PetscOptionsRealArray("-ts_adapt_glee_clip","Admissible decrease/increase in step size","",glee->clip,&two,&set);CHKERRQ(ierr);
  if (set && (two != 2 || glee->clip[0] > glee->clip[1])) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Must give exactly two values to -ts_adapt_glee_clip");
  ierr = PetscOptionsReal("-ts_adapt_glee_safety","Safety factor relative to target error","",glee->safety,&glee->safety,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_glee_reject_safety","Extra safety factor to apply if the last step was rejected","",glee->reject_safety,&glee->reject_safety,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_glee_always_accept","Always accept the step regardless of whether local truncation error meets goal","",glee->always_accept,&glee->always_accept,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptView_GLEE"
static PetscErrorCode TSAdaptView_GLEE(TSAdapt adapt,PetscViewer viewer)
{
  TSAdapt_GLEE  *glee = (TSAdapt_GLEE*)adapt->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (glee->always_accept) {ierr = PetscViewerASCIIPrintf(viewer,"  GLEE: always accepting steps\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  GLEE: clip fastest decrease %g, fastest increase %g\n",(double)glee->clip[0],(double)glee->clip[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  GLEE: safety factor %g, extra factor after step rejection %g\n",(double)glee->safety,(double)glee->reject_safety);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptCreate_GLEE"
/*MC
   TSADAPTGLEE - GLEE adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_GLEE(TSAdapt adapt)
{
  PetscErrorCode ierr;
  TSAdapt_GLEE  *a;

  PetscFunctionBegin;
  ierr                       = PetscNewLog(adapt,&a);CHKERRQ(ierr);
  adapt->data                = (void*)a;
  adapt->ops->choose         = TSAdaptChoose_GLEE;
  adapt->ops->setfromoptions = TSAdaptSetFromOptions_GLEE;
  adapt->ops->destroy        = TSAdaptDestroy_GLEE;
  adapt->ops->view           = TSAdaptView_GLEE;

  a->clip[0]       = 0.1;
  a->clip[1]       = 10.;
  a->safety        = 0.9;
  a->reject_safety = 0.5;
  a->always_accept = PETSC_FALSE;
  PetscFunctionReturn(0);
}
