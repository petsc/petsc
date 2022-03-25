
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
#if defined(PETSC_HAVE_METIS)
PETSC_INTERN PetscErrorCode MatGetOrdering_METISND(Mat,MatOrderingType,IS*,IS*);
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
  PetscFunctionBegin;
  if (MatOrderingRegisterAllCalled) PetscFunctionReturn(0);
  MatOrderingRegisterAllCalled = PETSC_TRUE;

  PetscCall(MatOrderingRegister(MATORDERINGNATURAL,  MatGetOrdering_Natural));
  PetscCall(MatOrderingRegister(MATORDERINGND,       MatGetOrdering_ND));
  PetscCall(MatOrderingRegister(MATORDERING1WD,      MatGetOrdering_1WD));
  PetscCall(MatOrderingRegister(MATORDERINGRCM,      MatGetOrdering_RCM));
  PetscCall(MatOrderingRegister(MATORDERINGQMD,      MatGetOrdering_QMD));
  PetscCall(MatOrderingRegister(MATORDERINGROWLENGTH,MatGetOrdering_RowLength));
#if defined(PETSC_HAVE_SUPERLU_DIST)
  PetscCall(MatOrderingRegister(MATORDERINGWBM,      MatGetOrdering_WBM));
#endif
  PetscCall(MatOrderingRegister(MATORDERINGSPECTRAL, MatGetOrdering_Spectral));
#if defined(PETSC_HAVE_SUITESPARSE)
  PetscCall(MatOrderingRegister(MATORDERINGAMD,      MatGetOrdering_AMD));
#endif
#if defined(PETSC_HAVE_METIS)
  PetscCall(MatOrderingRegister(MATORDERINGMETISND,  MatGetOrdering_METISND));
#endif
  PetscFunctionReturn(0);
}
