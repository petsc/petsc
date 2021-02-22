
#include <petsc/private/matimpl.h>     /*I       "petscmat.h"   I*/

PETSC_INTERN PetscErrorCode MatGetOrdering_Natural(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_ND(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_1WD(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_QMD(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_RCM(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_RowLength(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_DSC(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_WBM(Mat,MatOrderingType,IS*,IS*);
PETSC_INTERN PetscErrorCode MatGetOrdering_Spectral(Mat,MatOrderingType,IS*,IS*);
#if defined(PETSC_HAVE_SUITESPARSE)
PETSC_INTERN PetscErrorCode MatGetOrdering_AMD(Mat,MatOrderingType,IS*,IS*);
#endif

/*@C
  MatOrderingRegisterAll - Registers all of the matrix
  reordering routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and
  modify it to incorporate a call to MatReorderRegister() for
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and comment out
  those orderigs you do not wish to include.  Make sure that the
  replacement routine is linked before libpetscmat.a.

.seealso: MatOrderingRegister()
@*/
PetscErrorCode  MatOrderingRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatOrderingRegisterAllCalled) PetscFunctionReturn(0);
  MatOrderingRegisterAllCalled = PETSC_TRUE;

  ierr = MatOrderingRegister(MATORDERINGNATURAL,  MatGetOrdering_Natural);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGND,       MatGetOrdering_ND);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERING1WD,      MatGetOrdering_1WD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGRCM,      MatGetOrdering_RCM);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGQMD,      MatGetOrdering_QMD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGROWLENGTH,MatGetOrdering_RowLength);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUPERLU_DIST)
  ierr = MatOrderingRegister(MATORDERINGWBM,      MatGetOrdering_WBM);CHKERRQ(ierr);
#endif
  ierr = MatOrderingRegister(MATORDERINGSPECTRAL, MatGetOrdering_Spectral);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUITESPARSE)
  ierr = MatOrderingRegister(MATORDERINGAMD,      MatGetOrdering_AMD);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

