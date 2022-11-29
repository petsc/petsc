
#include <petscmat.h>
#include <petsc/private/matimpl.h>

PETSC_EXTERN PetscErrorCode MatColoringCreate_JP(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_Greedy(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_Power(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_Natural(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_SL(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_ID(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_LF(MatColoring);

/*@C
  MatColoringRegisterAll - Registers all of the matrix coloring routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and
  modify it to incorporate a call to `MatColoringRegister()` for
  the new method, after the current list.

 .seealso: `MatColoring`, `MatColoringRegister()`, `MatColoringRegisterDestroy()`
 @*/
PetscErrorCode MatColoringRegisterAll(void)
{
  PetscFunctionBegin;
  if (MatColoringRegisterAllCalled) PetscFunctionReturn(0);
  MatColoringRegisterAllCalled = PETSC_TRUE;
  PetscCall(MatColoringRegister(MATCOLORINGJP, MatColoringCreate_JP));
  PetscCall(MatColoringRegister(MATCOLORINGGREEDY, MatColoringCreate_Greedy));
  PetscCall(MatColoringRegister(MATCOLORINGPOWER, MatColoringCreate_Power));
  PetscCall(MatColoringRegister(MATCOLORINGNATURAL, MatColoringCreate_Natural));
  PetscCall(MatColoringRegister(MATCOLORINGSL, MatColoringCreate_SL));
  PetscCall(MatColoringRegister(MATCOLORINGID, MatColoringCreate_ID));
  PetscCall(MatColoringRegister(MATCOLORINGLF, MatColoringCreate_LF));
  PetscFunctionReturn(0);
}
