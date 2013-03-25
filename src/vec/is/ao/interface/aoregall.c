
#include <../src/vec/is/ao/aoimpl.h>      /*I "petscao.h" I*/
PETSC_EXTERN PetscErrorCode AOCreate_Basic(AO ao);
PETSC_EXTERN PetscErrorCode AOCreate_MemoryScalable(AO ao);

#undef __FUNCT__
#define __FUNCT__ "AORegisterAll"
/*@C
  AORegisterAll - Registers all of the application ordering components in the AO package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: AO, register, all
.seealso:  AORegister(), AORegisterDestroy(), AORegister()
@*/
PetscErrorCode  AORegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  AORegisterAllCalled = PETSC_TRUE;

  ierr = AORegister(AOBASIC,           path, "AOCreate_Basic",          AOCreate_Basic);CHKERRQ(ierr);
  ierr = AORegister(AOMEMORYSCALABLE,  path, "AOCreate_MemoryScalable", AOCreate_MemoryScalable);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
