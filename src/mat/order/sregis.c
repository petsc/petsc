
#include <petsc-private/matimpl.h>     /*I       "petscmat.h"   I*/

PETSC_EXTERN PetscErrorCode MatGetOrdering_Natural(Mat,MatOrderingType,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatGetOrdering_ND(Mat,MatOrderingType,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatGetOrdering_1WD(Mat,MatOrderingType,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatGetOrdering_QMD(Mat,MatOrderingType,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatGetOrdering_RCM(Mat,MatOrderingType,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatGetOrdering_RowLength(Mat,MatOrderingType,IS*,IS*);
PETSC_EXTERN PetscErrorCode MatGetOrdering_DSC(Mat,MatOrderingType,IS*,IS*);
#if defined(PETSC_HAVE_UMFPACK)
PETSC_EXTERN PetscErrorCode MatGetOrdering_AMD(Mat,MatOrderingType,IS*,IS*);
#endif

#undef __FUNCT__
#define __FUNCT__ "MatOrderingRegisterAll"
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

.keywords: matrix, reordering, register, all

.seealso: MatOrderingRegister(), MatOrderingRegisterDestroy()
@*/
PetscErrorCode  MatOrderingRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatOrderingRegisterAllCalled = PETSC_TRUE;

  ierr = MatOrderingRegister(MATORDERINGNATURAL,  path,"MatGetOrdering_Natural"  ,MatGetOrdering_Natural);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGND,       path,"MatGetOrdering_ND"       ,MatGetOrdering_ND);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERING1WD,      path,"MatGetOrdering_1WD"      ,MatGetOrdering_1WD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGRCM,      path,"MatGetOrdering_RCM"      ,MatGetOrdering_RCM);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGQMD,      path,"MatGetOrdering_QMD"      ,MatGetOrdering_QMD);CHKERRQ(ierr);
  ierr = MatOrderingRegister(MATORDERINGROWLENGTH,path,"MatGetOrdering_RowLength",MatGetOrdering_RowLength);CHKERRQ(ierr);
#if defined(PETSC_HAVE_UMFPACK)
  ierr = MatOrderingRegister(MATORDERINGAMD,      path,"MatGetOrdering_AMD",MatGetOrdering_AMD);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

