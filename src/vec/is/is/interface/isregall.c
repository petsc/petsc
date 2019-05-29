
#include <petsc/private/isimpl.h>     /*I  "petscis.h"  I*/
PETSC_EXTERN PetscErrorCode ISCreate_General(IS);
PETSC_EXTERN PetscErrorCode ISCreate_Stride(IS);
PETSC_EXTERN PetscErrorCode ISCreate_Block(IS);

/*@C
  ISRegisterAll - Registers all of the index set components in the IS package.

  Not Collective

  Level: advanced

.seealso:  ISRegister()
@*/
PetscErrorCode  ISRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ISRegisterAllCalled) PetscFunctionReturn(0);
  ISRegisterAllCalled = PETSC_TRUE;

  ierr = ISRegister(ISGENERAL, ISCreate_General);CHKERRQ(ierr);
  ierr = ISRegister(ISSTRIDE,  ISCreate_Stride);CHKERRQ(ierr);
  ierr = ISRegister(ISBLOCK,   ISCreate_Block);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

