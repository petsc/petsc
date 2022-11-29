#include <petsc/private/characteristicimpl.h> /*I "petsccharacteristic.h" I*/

PETSC_EXTERN PetscErrorCode CharacteristicCreate_DA(Characteristic);

/*@C
  CharacteristicRegisterAll - Registers all of the methods in the `Characteristic` package.

  Not Collective

  Level: advanced

.seealso: [](chapter_ts), `CharacteristicRegisterDestroy()`
@*/
PetscErrorCode CharacteristicRegisterAll(void)
{
  PetscFunctionBegin;
  if (CharacteristicRegisterAllCalled) PetscFunctionReturn(0);
  CharacteristicRegisterAllCalled = PETSC_TRUE;

  PetscCall(CharacteristicRegister(CHARACTERISTICDA, CharacteristicCreate_DA));
  PetscFunctionReturn(0);
}
