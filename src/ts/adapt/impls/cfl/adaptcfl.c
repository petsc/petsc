#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

static PetscErrorCode TSAdaptChoose_CFL(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  PetscErrorCode  ierr;
  PetscReal       hcfl,cfltimestep,ccfl;
  PetscInt        ncandidates;
  const PetscReal *ccflarray;

  PetscFunctionBegin;
  ierr = TSGetCFLTime(ts,&cfltimestep);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesGet(adapt,&ncandidates,NULL,NULL,&ccflarray,NULL);CHKERRQ(ierr);
  ccfl = (ncandidates > 0) ? ccflarray[0] : 1.0;

  if (!adapt->always_accept) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_SUP,"Step rejection not implemented. The CFL implementation is incomplete/unusable");

  /* Determine whether the step is accepted of rejected */
  *accept = PETSC_TRUE;
  if (h > cfltimestep * ccfl) {
    if (adapt->always_accept) {
      ierr = PetscInfo3(adapt,"Step length %g with scheme of CFL coefficient %g did not satisfy user-provided CFL constraint %g, proceeding anyway\n",(double)h,(double)ccfl,(double)cfltimestep);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo3(adapt,"Step length %g with scheme of CFL coefficient %g did not satisfy user-provided CFL constraint %g, step REJECTED\n",(double)h,(double)ccfl,(double)cfltimestep);CHKERRQ(ierr);
      *accept  = PETSC_FALSE;
    }
  }

  /* The optimal new step based purely on CFL constraint for this step. */
  hcfl = adapt->safety * cfltimestep * ccfl;
  if (hcfl < adapt->dt_min) {
    ierr = PetscInfo4(adapt,"Cannot satisfy CFL constraint %g (with %g safety) at minimum time step %g with method coefficient %g, proceding anyway\n",(double)cfltimestep,(double)adapt->safety,(double)adapt->dt_min,(double)ccfl);CHKERRQ(ierr);
  }

  *next_sc = 0;
  *next_h  = PetscClipInterval(hcfl,adapt->dt_min,adapt->dt_max);
  *wlte    = -1;   /* Weighted local truncation error was not evaluated */
  *wltea   = -1;   /* Weighted absolute local truncation error was not evaluated */
  *wlter   = -1;   /* Weighted relative local truncation error was not evaluated */
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTCFL - CFL adaptive controller for time stepping

   Level: intermediate

.seealso: TS, TSAdapt, TSGetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_CFL(TSAdapt adapt)
{
  PetscFunctionBegin;
  adapt->ops->choose = TSAdaptChoose_CFL;
  PetscFunctionReturn(0);
}
