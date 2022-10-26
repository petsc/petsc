
#include <../src/vec/is/ao/aoimpl.h> /*I "petscao.h" I*/
PETSC_EXTERN PetscErrorCode AOCreate_Basic(AO ao);
PETSC_EXTERN PetscErrorCode AOCreate_MemoryScalable(AO ao);

/*@C
  AORegisterAll - Registers all of the application ordering components in the `AO` package.

  Not Collective

  Level: advanced

.seealso: `AO`, `AOType`, `AORegister()`, `AORegisterDestroy()`
@*/
PetscErrorCode AORegisterAll(void)
{
  PetscFunctionBegin;
  if (AORegisterAllCalled) PetscFunctionReturn(0);
  AORegisterAllCalled = PETSC_TRUE;

  PetscCall(AORegister(AOBASIC, AOCreate_Basic));
  PetscCall(AORegister(AOMEMORYSCALABLE, AOCreate_MemoryScalable));
  PetscFunctionReturn(0);
}
