#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  PetscBool always_accept;
  PetscReal clip[2];            /* admissible decrease/increase factors */
  PetscReal safety;             /* safety factor relative to target error */
  PetscReal reject_safety;      /* extra safety factor if the last step was rejected */
  Vec       X;
  Vec       Y;
} TSAdapt_GLEE;

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_GLEE"
static PetscErrorCode TSAdaptChoose_GLEE(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte)
{
  TSAdapt_GLEE  *glee = (TSAdapt_GLEE*)adapt->data;
  TSType         time_scheme;      /* Type of time-integration scheme        */
  PetscErrorCode ierr;
  Vec            X,Y,Z;
  PetscReal      enorm,enorma,enormr,hfac_lte,h_lte,safety;
  PetscInt       order,order_add,stepno;

  PetscFunctionBegin;
  ierr = TSGetTimeStepNumber(ts,&stepno);CHKERRQ(ierr);

  safety = glee->safety;
  ierr = TSGetType(ts,&time_scheme);CHKERRQ(ierr);
  if (!strcmp(time_scheme,TSGLEE)){
    /* the method is of GLEE type */
    order_add=1; /* typically same order estimates */
    if (!glee->X) {
      ierr = TSGetSolution(ts,&Z);CHKERRQ(ierr);
      ierr = VecDuplicate(Z,&glee->X);CHKERRQ(ierr);
    }
    X    = glee->X;
    if (!glee->Y) {ierr = VecDuplicate(X,&glee->Y);CHKERRQ(ierr);}
    Y    = glee->Y;
    ierr = TSGetTimeError(ts,-1,&X);CHKERRQ(ierr);
    ierr = TSGetTimeError(ts, 0,&Y);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,X,Y,adapt->wnormtype,&enorm,&enorma,&enormr);CHKERRQ(ierr);
  } else {
    /* the method is NOT of GLEE type */
    order_add=0; /* typically lower order estimates */
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!glee->Y) {ierr = VecDuplicate(X,&glee->Y);CHKERRQ(ierr);}
    Y     = glee->Y;
    order = adapt->candidates.order[0];
    ierr  = TSEvaluateStep(ts,order-1,Y,NULL);CHKERRQ(ierr);
    ierr  = TSErrorWeightedNorm(ts,X,Y,adapt->wnormtype,&enorm,&enorma,&enormr);CHKERRQ(ierr);
  }
  if (enorm > 1.) {
    if (!*accept) safety *= glee->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (glee->always_accept) {
      ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g because always_accept is set\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else {
      ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, rejecting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
    }
  } else {
    ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
    *accept = PETSC_TRUE;
  }

  /* The optimal new step based purely on local truncation error for this step. */
  if (enorm == 0.0) {
    hfac_lte = safety * PETSC_INFINITY;
  } else {
    hfac_lte = safety * PetscPowReal(enorm,-1./(order+order_add));
  }
  h_lte    = h * PetscClipInterval(hfac_lte,glee->clip[0],glee->clip[1]);

  *next_sc = 0;
  *next_h  = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  *wlte    = enorm;
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
