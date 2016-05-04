#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  PetscBool always_accept;
  PetscReal clip[2];            /* admissible decrease/increase factors */
  PetscReal safety;             /* safety factor relative to target error */
  PetscReal reject_safety;      /* extra safety factor if the last step was rejected */
  Vec       Y;
} TSAdapt_Basic;

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_Basic"
static PetscErrorCode TSAdaptChoose_Basic(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscInt       order  = PETSC_DECIDE;
  PetscReal      enorm  = -1;
  PetscReal      safety = basic->safety;
  PetscReal      hfac_lte,h_lte;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *next_sc = 0; /* Reuse the same order scheme */

  if (ts->ops->evaluatewlte) {
    ierr = TSEvaluateWLTE(ts,adapt->wnormtype,&order,&enorm);CHKERRQ(ierr);
    if (enorm >= 0 && order < 1) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed error order %D must be positive",order);
  } else if (ts->ops->evaluatestep) {
    if (adapt->candidates.n < 1) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"No candidate has been registered");
    if (!adapt->candidates.inuse_set) SETERRQ1(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
    if (!basic->Y) {ierr = VecDuplicate(ts->vec_sol,&basic->Y);CHKERRQ(ierr);}
    order = adapt->candidates.order[0];
    ierr = TSEvaluateStep(ts,order-1,basic->Y,NULL);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,ts->vec_sol,basic->Y,adapt->wnormtype,&enorm);CHKERRQ(ierr);
  }

  if (enorm < 0) {
    *accept  = PETSC_TRUE;
    *next_h  = h;            /* Reuse the old step */
    *wlte    = -1;           /* Weighted local truncation error was not evaluated */
    PetscFunctionReturn(0);
  }

  /* Determine whether the step is accepted of rejected */
  if (enorm > 1) {
    if (!*accept) safety *= basic->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (basic->always_accept) {
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
  h_lte = h * PetscClipInterval(hfac_lte,basic->clip[0],basic->clip[1]);

  *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  *wlte   = enorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptReset_Basic"
static PetscErrorCode TSAdaptReset_Basic(TSAdapt adapt)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&basic->Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptDestroy_Basic"
static PetscErrorCode TSAdaptDestroy_Basic(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptReset_Basic(adapt);CHKERRQ(ierr);
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetFromOptions_Basic"
static PetscErrorCode TSAdaptSetFromOptions_Basic(PetscOptionItems *PetscOptionsObject,TSAdapt adapt)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscErrorCode ierr;
  PetscInt       two;
  PetscBool      set;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Basic adaptive controller options");CHKERRQ(ierr);
  two  = 2;
  ierr = PetscOptionsRealArray("-ts_adapt_basic_clip","Admissible decrease/increase factor in step size","TSAdaptBasicSetClip",basic->clip,&two,&set);CHKERRQ(ierr);
  if (set && (two != 2)) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Must give exactly two values to -ts_adapt_basic_clip");
  if (set) {ierr = TSAdaptBasicSetClip(adapt,basic->clip[0],basic->clip[1]);CHKERRQ(ierr);}
  ierr = PetscOptionsReal("-ts_adapt_basic_safety","Safety factor relative to target error","",basic->safety,&basic->safety,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_basic_reject_safety","Extra safety factor to apply if the last step was rejected","",basic->reject_safety,&basic->reject_safety,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_basic_always_accept","Always accept the step regardless of whether local truncation error meets goal","",basic->always_accept,&basic->always_accept,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptView_Basic"
static PetscErrorCode TSAdaptView_Basic(TSAdapt adapt,PetscViewer viewer)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (basic->always_accept) {ierr = PetscViewerASCIIPrintf(viewer,"  Basic: always accepting steps\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"  Basic: clip fastest decrease %g, fastest increase %g\n",(double)basic->clip[0],(double)basic->clip[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Basic: safety factor %g, extra factor after step rejection %g\n",(double)basic->safety,(double)basic->reject_safety);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptCreate_Basic"
/*MC
   TSADAPTBASIC - Basic adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Basic(TSAdapt adapt)
{
  PetscErrorCode ierr;
  TSAdapt_Basic  *a;

  PetscFunctionBegin;
  ierr                       = PetscNewLog(adapt,&a);CHKERRQ(ierr);
  adapt->data                = (void*)a;
  adapt->ops->choose         = TSAdaptChoose_Basic;
  adapt->ops->setfromoptions = TSAdaptSetFromOptions_Basic;
  adapt->ops->destroy        = TSAdaptDestroy_Basic;
  adapt->ops->view           = TSAdaptView_Basic;

  a->clip[0]       = 0.1;
  a->clip[1]       = 10.;
  a->safety        = 0.9;
  a->reject_safety = 0.5;
  a->always_accept = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptBasicSetClip"
/*@
   TSAdaptBasicSetClip - Sets the admissible decrease/increase factor in step size

   Collective on TSAdapt

   Input Arguments:
+  adapt - adaptive controller context
.  low - admissible decrease factor
+  high - admissible increase factor

   Level: intermediate

.seealso: TSAdaptChoose()
@*/
PetscErrorCode TSAdaptBasicSetClip(TSAdapt adapt,PetscReal low,PetscReal high)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscBool      isbasic;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (low  != PETSC_DEFAULT && low  < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Decrease factor %g must be non negative",(double)low);
  if (low  != PETSC_DEFAULT && low  > 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Decrease factor %g must be less than one",(double)low);
  if (high != PETSC_DEFAULT && high < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Increase factor %g must be geather than one",(double)high);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTBASIC,&isbasic);CHKERRQ(ierr);
  if (isbasic) {
    if (low  != PETSC_DEFAULT) basic->clip[0] = low;
    if (high != PETSC_DEFAULT) basic->clip[1] = high;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptBasicGetClip"
/*@
   TSAdaptBasicGetClip - Gets the admissible decrease/increase factor in step size

   Collective on TSAdapt

   Input Arguments:
.  adapt - adaptive controller context

   Ouput Arguments:
+  low - optional, admissible decrease factor
-  high - optional, admissible increase factor

   Level: intermediate

.seealso: TSAdaptChoose()
@*/
PetscErrorCode TSAdaptBasicGetClip(TSAdapt adapt,PetscReal *low,PetscReal *high)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscBool      isbasic;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (low)  PetscValidRealPointer(low,2);
  if (high) PetscValidRealPointer(high,3);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTBASIC,&isbasic);CHKERRQ(ierr);
  if (isbasic) {
    if (low)  *low  = basic->clip[0];
    if (high) *high = basic->clip[1];
  } else {
    if (low)  *low  = 0;
    if (high) *high = PETSC_MAX_REAL;
  }
  PetscFunctionReturn(0);
}
