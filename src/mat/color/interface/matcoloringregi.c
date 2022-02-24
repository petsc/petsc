
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
  MatColoringRegisterAll - Registers all of the matrix Coloring routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and
  modify it to incorporate a call to MatColoringRegister() for
  the new method, after the current list.

 .seealso: MatColoringRegister(), MatColoringRegisterDestroy()
 @*/
PetscErrorCode  MatColoringRegisterAll(void)
{
  PetscFunctionBegin;
  if (MatColoringRegisterAllCalled) PetscFunctionReturn(0);
  MatColoringRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(MatColoringRegister(MATCOLORINGJP,MatColoringCreate_JP));
  CHKERRQ(MatColoringRegister(MATCOLORINGGREEDY,MatColoringCreate_Greedy));
  CHKERRQ(MatColoringRegister(MATCOLORINGPOWER,MatColoringCreate_Power));
  CHKERRQ(MatColoringRegister(MATCOLORINGNATURAL,MatColoringCreate_Natural));
  CHKERRQ(MatColoringRegister(MATCOLORINGSL,MatColoringCreate_SL));
  CHKERRQ(MatColoringRegister(MATCOLORINGID,MatColoringCreate_ID));
  CHKERRQ(MatColoringRegister(MATCOLORINGLF,MatColoringCreate_LF));
  PetscFunctionReturn(0);
}
