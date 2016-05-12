#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  PetscReal safety;         /* safety factor relative to target CFL constraint */
  PetscBool always_accept;
} TSAdapt_CFL;

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_CFL"
static PetscErrorCode TSAdaptChoose_CFL(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte)
{
  TSAdapt_CFL     *cfl = (TSAdapt_CFL*)adapt->data;
  PetscErrorCode  ierr;
  PetscReal       hcfl,cfltimestep,ccfl;
  PetscInt        ncandidates;
  const PetscReal *ccflarray;

  PetscFunctionBegin;
  ierr = TSGetCFLTime(ts,&cfltimestep);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesGet(adapt,&ncandidates,NULL,NULL,&ccflarray,NULL);CHKERRQ(ierr);
  ccfl = (ncandidates > 0) ? ccflarray[0] : 1.0;

  /* Determine whether the step is accepted of rejected */
  *accept = PETSC_TRUE;
  if (h > cfltimestep * ccfl) {
    if (cfl->always_accept) {
      ierr = PetscInfo3(adapt,"Step length %g with scheme of CFL coefficient %g did not satisfy user-provided CFL constraint %g, proceeding anyway\n",(double)h,(double)ccfl,(double)cfltimestep);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo3(adapt,"Step length %g with scheme of CFL coefficient %g did not satisfy user-provided CFL constraint %g, step REJECTED\n",(double)h,(double)ccfl,(double)cfltimestep);CHKERRQ(ierr);
      *accept  = PETSC_FALSE;
    }
  }

  /* The optimal new step based purely on CFL constraint for this step. */
  hcfl = cfl->safety * cfltimestep * ccfl;
  if (hcfl < adapt->dt_min) {
    ierr = PetscInfo4(adapt,"Cannot satisfy CFL constraint %g (with %g safety) at minimum time step %g with method coefficient %g, proceding anyway\n",(double)cfltimestep,(double)cfl->safety,(double)adapt->dt_min,(double)ccfl);CHKERRQ(ierr);
  }

  *next_sc = 0;
  *next_h  = PetscClipInterval(hcfl,adapt->dt_min,adapt->dt_max);
  *wlte    = -1;  /* Weighted local truncation error was not evaluated */
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
static PetscErrorCode TSAdaptSetFromOptions_CFL(PetscOptionItems *PetscOptionsObject,TSAdapt adapt)
{
  TSAdapt_CFL    *cfl = (TSAdapt_CFL*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"CFL adaptive controller options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_adapt_cfl_safety","Safety factor relative to target CFL constraint","",cfl->safety,&cfl->safety,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_adapt_cfl_always_accept","Always accept the step regardless of whether CFL constraint meets goal","",cfl->always_accept,&cfl->always_accept,NULL);CHKERRQ(ierr);
  /* The TS implementations do not currently communicate CFL information to the controller.  There is a placeholder, but
   * we do not believe it to provide sufficiently rich information.  That means the CFL adaptor is incomplete and
   * unusable.  Do not delete the guard below unless you have finished the implementation. */
  if (!cfl->always_accept) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_SUP,"Step rejection not implemented. The CFL implementation is incomplete/unusable");
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptCreate_CFL"
/*MC
   TSADAPTCFL - CFL adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_CFL(TSAdapt adapt)
{
  PetscErrorCode ierr;
  TSAdapt_CFL    *a;

  PetscFunctionBegin;
  ierr = PetscNewLog(adapt,&a);CHKERRQ(ierr);
  adapt->data = (void*)a;

  adapt->ops->choose         = TSAdaptChoose_CFL;
  adapt->ops->setfromoptions = TSAdaptSetFromOptions_CFL;
  adapt->ops->destroy        = TSAdaptDestroy_CFL;

  a->safety        = 0.9;
  a->always_accept = PETSC_FALSE;
  PetscFunctionReturn(0);
}
