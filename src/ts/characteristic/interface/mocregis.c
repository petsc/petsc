#include <petsc-private/characteristicimpl.h>  /*I "petsccharacteristic.h" I*/

EXTERN_C_BEGIN
extern PetscErrorCode CharacteristicCreate_DA(Characteristic);
EXTERN_C_END

/*
    This is used by CharacteristicSetType() to make sure that at least one
    CharacteristicRegisterAll() is called. In general, if there is more than one
    DLL, then CharacteristicRegisterAll() may be called several times.
*/
extern PetscBool  CharacteristicRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "CharacteristicRegisterAll"
/*@C
  CharacteristicRegisterAll - Registers all of the Krylov subspace methods in the Characteristic package.

  Not Collective

  Level: advanced

.keywords: Characteristic, register, all

.seealso:  CharacteristicRegisterDestroy()
@*/
PetscErrorCode CharacteristicRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CharacteristicRegisterAllCalled = PETSC_TRUE;

  ierr = CharacteristicRegisterDynamic(CHARACTERISTICDA, path, "CharacteristicCreate_DA", CharacteristicCreate_DA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
