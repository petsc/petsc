
#include <petscmat.h>

PETSC_EXTERN PetscErrorCode MatColoringCreate_JP(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_MIS(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_Natural(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_SL(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_ID(MatColoring);
PETSC_EXTERN PetscErrorCode MatColoringCreate_LF(MatColoring);

#undef __FUNCT__
#define __FUNCT__ "MatColoringRegisterAll"
/*@C
  MatColoringRegisterAll - Registers all of the matrix Coloring routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and
  modify it to incorporate a call to MatColoringRegister() for
  the new method, after the current list.

 .keywords: matrix, coloring, register, all

 .seealso: MatColoringRegister(), MatColoringRegisterDestroy()
 @*/
PetscErrorCode  MatColoringRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatColoringRegisterAllCalled = PETSC_TRUE;
  ierr = MatColoringRegister(MATCOLORINGJP,MatColoringCreate_JP);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGMIS,MatColoringCreate_MIS);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGNATURAL,MatColoringCreate_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGSL,MatColoringCreate_SL);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGID,MatColoringCreate_ID);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGLF,MatColoringCreate_LF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
