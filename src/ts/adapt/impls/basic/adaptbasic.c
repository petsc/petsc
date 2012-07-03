#include <petsc-private/tsimpl.h> /*I "petscts.h" I*/

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
  TSAdapt_Basic     *basic = (TSAdapt_Basic*)adapt->data;
  PetscErrorCode    ierr;
  Vec               X,Y;
  PetscReal         enorm,hfac_lte,h_lte,safety;
  PetscInt          order,stepno;

  PetscFunctionBegin;
  ierr = TSGetTimeStepNumber(ts,&stepno);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  if (!basic->Y) {ierr = VecDuplicate(X,&basic->Y);CHKERRQ(ierr);}
  Y = basic->Y;
  order = adapt->candidates.order[0];
  ierr = TSEvaluateStep(ts,order-1,Y,PETSC_NULL);CHKERRQ(ierr);

  safety = basic->safety;
  ierr = TSErrorNormWRMS(ts,Y,&enorm);CHKERRQ(ierr);
  if (enorm > 1.) {
    if (!*accept) safety *= basic->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %G, accepting because step size %G is at minimum\n",enorm,h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (basic->always_accept) {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %G, accepting step of size %G because always_accept is set\n",enorm,h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else {
      ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %G, rejecting step of size %G\n",enorm,h);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
    }
  } else {
    ierr = PetscInfo2(adapt,"Estimated scaled local truncation error %G, accepting step of size %G\n",enorm,h);CHKERRQ(ierr);
    *accept = PETSC_TRUE;
  }

  /* The optimal new step based purely on local truncation error for this step. */
  hfac_lte = safety * PetscRealPart(PetscPowScalar((PetscScalar)enorm,(PetscReal)(-1./order)));
  h_lte = h * PetscClipInterval(hfac_lte,basic->clip[0],basic->clip[1]);

  *next_sc = 0;
  *next_h = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  *wlte = enorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptDestroy_Basic"
static PetscErrorCode TSAdaptDestroy_Basic(TSAdapt adapt)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&basic->Y);CHKERRQ(ierr);
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetFromOptions_Basic"
static PetscErrorCode TSAdaptSetFromOptions_Basic(TSAdapt adapt)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscErrorCode ierr;
  PetscInt       two;
  PetscBool      set;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Basic adaptive controller options");CHKERRQ(ierr);
  two = 2;
  ierr = PetscOptionsRealArray("-ts_adapt_basic_clip","Admissible decrease/increase in step size","",basic->clip,&two,&set);CHKERRQ(ierr);
  if (set && (two != 2 || basic->clip[0] > basic->clip[1]))
    SETERRQ(((PetscObject)adapt)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Must give exactly two values to -ts_adapt_basic_clip");
  ierr = PetscOptionsReal("-ts_adapt_basic_safety","Safety factor relative to target error","",basic->safety,&basic->safety,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_basic_reject_safety","Extra safety factor to apply if the last step was rejected","",basic->reject_safety,&basic->reject_safety,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_basic_always_accept","Always accept the step regardless of whether local truncation error meets goal","",basic->always_accept,&basic->always_accept,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSAdaptCreate_Basic"
/*MC
   TSADAPTBASIC - Basic adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PetscErrorCode TSAdaptCreate_Basic(TSAdapt adapt)
{
  PetscErrorCode ierr;
  TSAdapt_Basic *a;

  PetscFunctionBegin;
  ierr = PetscNewLog(adapt,TSAdapt_Basic,&a);CHKERRQ(ierr);
  adapt->data = (void*)a;
  adapt->ops->choose         = TSAdaptChoose_Basic;
  adapt->ops->setfromoptions = TSAdaptSetFromOptions_Basic;
  adapt->ops->destroy        = TSAdaptDestroy_Basic;

  a->clip[0]       = 0.1;
  a->clip[1]       = 10.;
  a->safety        = 0.9;
  a->reject_safety = 0.5;
  a->always_accept = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
