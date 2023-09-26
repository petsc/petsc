#include <petsc/private/isimpl.h> /*I  "petscis.h"  I*/
PETSC_INTERN PetscErrorCode ISCreate_General(IS);
PETSC_INTERN PetscErrorCode ISCreate_Stride(IS);
PETSC_INTERN PetscErrorCode ISCreate_Block(IS);

/*@C
  ISRegisterAll - Registers all of the index set components in the `IS` package.

  Not Collective

  Level: advanced

.seealso: [](sec_scatter), `IS`, `ISType`, `ISRegister()`
@*/
PetscErrorCode ISRegisterAll(void)
{
  PetscFunctionBegin;
  if (ISRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  ISRegisterAllCalled = PETSC_TRUE;

  PetscCall(ISRegister(ISGENERAL, ISCreate_General));
  PetscCall(ISRegister(ISSTRIDE, ISCreate_Stride));
  PetscCall(ISRegister(ISBLOCK, ISCreate_Block));
  PetscFunctionReturn(PETSC_SUCCESS);
}
