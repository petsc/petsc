#include <petsc/private/linesearchimpl.h> /*I  "petscsnes.h"  I*/

PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Basic(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_L2(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_CP(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_BT(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_NLEQERR(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Shell(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_NCGLinear(SNESLineSearch);

/*@C
   SNESLineSearchRegisterAll - Registers all of the nonlinear solver methods in the `SNESLineSearch` package.

   Not Collective

   Level: advanced

.seealso: `SNESLineSearchRegister()`, `SNESLineSearchRegisterDestroy()`
@*/
PetscErrorCode SNESLineSearchRegisterAll(void)
{
  PetscFunctionBegin;
  if (SNESLineSearchRegisterAllCalled) PetscFunctionReturn(0);
  SNESLineSearchRegisterAllCalled = PETSC_TRUE;
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHSHELL, SNESLineSearchCreate_Shell));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHBASIC, SNESLineSearchCreate_Basic));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHNONE, SNESLineSearchCreate_Basic));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHL2, SNESLineSearchCreate_L2));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHBT, SNESLineSearchCreate_BT));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHNLEQERR, SNESLineSearchCreate_NLEQERR));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHCP, SNESLineSearchCreate_CP));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHNCGLINEAR, SNESLineSearchCreate_NCGLinear));
  PetscFunctionReturn(0);
}
