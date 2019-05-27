
#include <../src/vec/is/ao/aoimpl.h>      /*I "petscao.h" I*/
PETSC_EXTERN PetscErrorCode AOCreate_Basic(AO ao);
PETSC_EXTERN PetscErrorCode AOCreate_MemoryScalable(AO ao);

/*@C
  AORegisterAll - Registers all of the application ordering components in the AO package.

  Not Collective

  Level: advanced

.seealso:  AORegister(), AORegisterDestroy()
@*/
PetscErrorCode  AORegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (AORegisterAllCalled) PetscFunctionReturn(0);
  AORegisterAllCalled = PETSC_TRUE;

  ierr = AORegister(AOBASIC,          AOCreate_Basic);CHKERRQ(ierr);
  ierr = AORegister(AOMEMORYSCALABLE, AOCreate_MemoryScalable);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
