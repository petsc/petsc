#include <petsc/private/linesearchimpl.h> /*I  "petscsnes.h"  I*/

PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_None(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Secant(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_CP(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_BT(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_NLEQERR(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Shell(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_NCGLinear(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Bisection(SNESLineSearch);

/*@C
  SNESLineSearchRegisterAll - Registers all of the nonlinear solver methods in the `SNESLineSearch` package.

  Not Collective

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchRegister()`, `SNESLineSearchRegisterDestroy()`
@*/
PetscErrorCode SNESLineSearchRegisterAll(void)
{
  PetscFunctionBegin;
  if (SNESLineSearchRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  SNESLineSearchRegisterAllCalled = PETSC_TRUE;
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHSHELL, SNESLineSearchCreate_Shell));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHNONE, SNESLineSearchCreate_None));
  PetscCall(SNESLineSearchRegister("basic", SNESLineSearchCreate_None)); // deprecated since version 3.26.0
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHSECANT, SNESLineSearchCreate_Secant));
  PetscCall(SNESLineSearchRegister("l2", SNESLineSearchCreate_Secant)); // deprecated since version 3.24.0
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHBT, SNESLineSearchCreate_BT));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHNLEQERR, SNESLineSearchCreate_NLEQERR));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHCP, SNESLineSearchCreate_CP));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHNCGLINEAR, SNESLineSearchCreate_NCGLinear));
  PetscCall(SNESLineSearchRegister(SNESLINESEARCHBISECTION, SNESLineSearchCreate_Bisection));
  PetscFunctionReturn(PETSC_SUCCESS);
}
