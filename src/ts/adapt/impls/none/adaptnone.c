#include <petsc-private/tsimpl.h> /*I "petscts.h" I*/

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_None"
static PetscErrorCode TSAdaptChoose_None(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte)
{

  PetscFunctionBegin;
  *accept = PETSC_TRUE;
  *next_sc = 0;                 /* Reuse the same order scheme */
  *next_h = h;                  /* Reuse the old step */
  *wlte = -1;                   /* Weighted local truncation error was not evaluated */
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSAdaptCreate_None"
/*MC
   TSADAPTNONE - Time stepping controller that always accepts the current step and does not change it

   Level: intermediate

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PetscErrorCode TSAdaptCreate_None(TSAdapt adapt)
{

  PetscFunctionBegin;
  adapt->ops->choose = TSAdaptChoose_None;
  PetscFunctionReturn(0);
}
EXTERN_C_END
