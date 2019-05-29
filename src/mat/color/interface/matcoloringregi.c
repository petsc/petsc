
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatColoringRegisterAllCalled) PetscFunctionReturn(0);
  MatColoringRegisterAllCalled = PETSC_TRUE;
  ierr = MatColoringRegister(MATCOLORINGJP,MatColoringCreate_JP);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGGREEDY,MatColoringCreate_Greedy);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGPOWER,MatColoringCreate_Power);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGNATURAL,MatColoringCreate_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGSL,MatColoringCreate_SL);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGID,MatColoringCreate_ID);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGLF,MatColoringCreate_LF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
