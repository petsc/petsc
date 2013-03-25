
#include <petsc-private/isimpl.h>     /*I  "petscis.h"  I*/
PETSC_EXTERN PetscErrorCode ISCreate_General(IS);
PETSC_EXTERN PetscErrorCode ISCreate_Stride(IS);
PETSC_EXTERN PetscErrorCode ISCreate_Block(IS);

#undef __FUNCT__
#define __FUNCT__ "ISRegisterAll"
/*@C
  ISRegisterAll - Registers all of the index set components in the IS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: IS, register, all
.seealso:  ISRegister(), ISRegisterDestroy(), ISRegister()
@*/
PetscErrorCode  ISRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ISRegisterAllCalled = PETSC_TRUE;

  ierr = ISRegister(ISGENERAL,     path, "ISCreate_General",    ISCreate_General);CHKERRQ(ierr);
  ierr = ISRegister(ISSTRIDE,      path, "ISCreate_Stride",     ISCreate_Stride);CHKERRQ(ierr);
  ierr = ISRegister(ISBLOCK,       path, "ISCreate_Block",      ISCreate_Block);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

