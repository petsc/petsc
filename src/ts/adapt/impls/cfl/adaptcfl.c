#include <petsc-private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  PetscBool always_accept;
  PetscReal safety;             /* safety factor relative to target error */
} TSAdapt_CFL;

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_CFL"
static PetscErrorCode TSAdaptChoose_CFL(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte)
{
  TSAdapt_CFL     *cfl = (TSAdapt_CFL*)adapt->data;
  PetscErrorCode  ierr;
  PetscReal       hcfl,cfltime;
  PetscInt        stepno,ncandidates;
  const PetscInt  *order;
  const PetscReal *ccfl;

  PetscFunctionBegin;
  ierr = TSGetTimeStepNumber(ts,&stepno);CHKERRQ(ierr);
  ierr = TSGetCFLTime(ts,&cfltime);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesGet(adapt,&ncandidates,&order,PETSC_NULL,&ccfl,PETSC_NULL);CHKERRQ(ierr);

  hcfl = cfl->safety * cfltime * ccfl[0];
  if (hcfl < adapt->dt_min) {
    ierr = PetscInfo4(adapt,"Cannot satisfy CFL constraint %G (with %G safety) at minimum time step %G with method coefficient %G, proceding anyway\n",cfltime,cfl->safety,adapt->dt_min,ccfl[0]);CHKERRQ(ierr);
  }

  if (h > cfltime * ccfl[0]) {
    if (cfl->always_accept) {
      ierr = PetscInfo3(adapt,"Step length %G with scheme of CFL coefficient %G did not satisfy user-provided CFL constraint %G, proceeding anyway\n",h,ccfl[0],cfltime);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo3(adapt,"Step length %G with scheme of CFL coefficient %G did not satisfy user-provided CFL constraint %G, step REJECTED\n",h,ccfl[0],cfltime);CHKERRQ(ierr);
      *next_sc = 0;
      *next_h = PetscClipInterval(hcfl,adapt->dt_min,adapt->dt_max);
      *accept = PETSC_FALSE;
    }
  }

  *next_sc = 0;
  *next_h = PetscClipInterval(hcfl,adapt->dt_min,adapt->dt_max);
  *accept = PETSC_TRUE;
  *wlte = -1;                   /* Weighted local truncation error was not evaluated */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptDestroy_CFL"
static PetscErrorCode TSAdaptDestroy_CFL(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptSetFromOptions_CFL"
static PetscErrorCode TSAdaptSetFromOptions_CFL(TSAdapt adapt)
{
  TSAdapt_CFL  *cfl = (TSAdapt_CFL*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("CFL adaptive controller options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_cfl_safety","Safety factor relative to target error","",cfl->safety,&cfl->safety,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_cfl_always_accept","Always accept the step regardless of whether local truncation error meets goal","",cfl->always_accept,&cfl->always_accept,PETSC_NULL);CHKERRQ(ierr);
  if (!cfl->always_accept) SETERRQ(((PetscObject)adapt)->comm,PETSC_ERR_SUP,"step rejection not implemented yet");
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSAdaptCreate_CFL"
/*MC
   TSADAPTCFL - CFL adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PetscErrorCode TSAdaptCreate_CFL(TSAdapt adapt)
{
  PetscErrorCode ierr;
  TSAdapt_CFL *a;

  PetscFunctionBegin;
  ierr = PetscNewLog(adapt,TSAdapt_CFL,&a);CHKERRQ(ierr);
  adapt->data = (void*)a;
  adapt->ops->choose         = TSAdaptChoose_CFL;
  adapt->ops->setfromoptions = TSAdaptSetFromOptions_CFL;
  adapt->ops->destroy        = TSAdaptDestroy_CFL;

  a->safety        = 0.9;
  a->always_accept = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
