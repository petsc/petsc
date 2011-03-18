
#include <../src/dm/ao/aoimpl.h>      /*I "petscao.h" I*/
EXTERN_C_BEGIN
extern PetscErrorCode AOCreate_Basic(AO ao);
extern PetscErrorCode AOCreate_BasicMemoryScalable(AO ao);
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "AORegisterAll"
/*@C
  AORegisterAll - Registers all of the application ordering components in the AO package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: AO, register, all
.seealso:  AORegister(), AORegisterDestroy(), AORegisterDynamic()
@*/
PetscErrorCode  AORegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  AORegisterAllCalled = PETSC_TRUE;

  ierr = AORegisterDynamic(AOBASIC,               path, "AOCreate_Basic",               AOCreate_Basic);CHKERRQ(ierr);
  ierr = AORegisterDynamic(AOBASICMEMORYSCALABLE, path, "AOCreate_BasicMemoryScalable", AOCreate_BasicMemoryScalable);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
