
#include <petsc/private/isimpl.h> /*I  "petscis.h"  I*/
PETSC_EXTERN PetscErrorCode ISCreate_General(IS);
PETSC_EXTERN PetscErrorCode ISCreate_Stride(IS);
PETSC_EXTERN PetscErrorCode ISCreate_Block(IS);

/*@C
  ISRegisterAll - Registers all of the index set components in the `IS` package.

  Not Collective

  Level: advanced

.seealso: [](sec_scatter), `IS`, `ISType`, `ISRegister()`
@*/
PetscErrorCode ISRegisterAll(void)
{
  PetscFunctionBegin;
  if (ISRegisterAllCalled) PetscFunctionReturn(0);
  ISRegisterAllCalled = PETSC_TRUE;

  PetscCall(ISRegister(ISGENERAL, ISCreate_General));
  PetscCall(ISRegister(ISSTRIDE, ISCreate_Stride));
  PetscCall(ISRegister(ISBLOCK, ISCreate_Block));
  PetscFunctionReturn(0);
}
