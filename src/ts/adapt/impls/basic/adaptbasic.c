#include <private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  PetscReal atol;               /* Absolute tolerance for errors committed in one time step */
  PetscReal rtol;               /* Relative tolerance */
  PetscReal rmin,rmax;          /* safety factors for increasing/decreasing step size */
  PetscReal nu;
} TSAdapt_Basic;

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_Basic_1"
static PetscErrorCode TSAdaptChoose_Basic_1(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept)
{
  PETSC_UNUSED TSAdapt_Basic *basic = (TSAdapt_Basic*)adapt->data;
  PETSC_UNUSED PetscErrorCode ierr;

  PetscFunctionBegin;
  *next_sc = 0;
  *next_h = h;
  *accept = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdaptDestroy_Basic"
static PetscErrorCode TSAdaptDestroy_Basic(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
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
  adapt->ops->choose = TSAdaptChoose_Basic_1;
  adapt->ops->destroy = TSAdaptDestroy_Basic;
  PetscFunctionReturn(0);
}
EXTERN_C_END
