
#include <petsc-private/isimpl.h>     /*I  "petscis.h"  I*/
EXTERN_C_BEGIN
extern PetscErrorCode  ISCreate_General(IS);
extern PetscErrorCode  ISCreate_Stride(IS);
extern PetscErrorCode  ISCreate_Block(IS);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "ISRegisterAll"
/*@C
  ISRegisterAll - Registers all of the index set components in the IS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: IS, register, all
.seealso:  ISRegister(), ISRegisterDestroy(), ISRegisterDynamic()
@*/
PetscErrorCode  ISRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ISRegisterAllCalled = PETSC_TRUE;

  ierr = ISRegisterDynamic(ISGENERAL,     path, "ISCreate_General",    ISCreate_General);CHKERRQ(ierr);
  ierr = ISRegisterDynamic(ISSTRIDE,      path, "ISCreate_Stride",     ISCreate_Stride);CHKERRQ(ierr);
  ierr = ISRegisterDynamic(ISBLOCK,       path, "ISCreate_Block",      ISCreate_Block);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

