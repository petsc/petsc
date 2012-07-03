
#include <petsc-private/matimpl.h>     /*I       "petscmat.h"   I*/

EXTERN_C_BEGIN
extern PetscErrorCode  MatGetOrdering_Natural(Mat,const MatOrderingType,IS*,IS*);
extern PetscErrorCode  MatGetOrdering_ND(Mat,const MatOrderingType,IS*,IS*);
extern PetscErrorCode  MatGetOrdering_1WD(Mat,const MatOrderingType,IS*,IS*);
extern PetscErrorCode  MatGetOrdering_QMD(Mat,const MatOrderingType,IS*,IS*);
extern PetscErrorCode  MatGetOrdering_RCM(Mat,const MatOrderingType,IS*,IS*);
extern PetscErrorCode  MatGetOrdering_RowLength(Mat,const MatOrderingType,IS*,IS*);
extern PetscErrorCode  MatGetOrdering_DSC(Mat,const MatOrderingType,IS*,IS*);
#if defined(PETSC_HAVE_UMFPACK)
extern PetscErrorCode  MatGetOrdering_AMD(Mat,const MatOrderingType,IS*,IS*);
#endif

EXTERN_C_END

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

.seealso: MatOrderingRegisterDynamic(), MatOrderingRegisterDestroy()
@*/
PetscErrorCode  MatOrderingRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatOrderingRegisterAllCalled = PETSC_TRUE;

  ierr = MatOrderingRegisterDynamic(MATORDERINGNATURAL,  path,"MatGetOrdering_Natural"  ,MatGetOrdering_Natural);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGND,       path,"MatGetOrdering_ND"       ,MatGetOrdering_ND);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERING1WD,      path,"MatGetOrdering_1WD"      ,MatGetOrdering_1WD);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGRCM,      path,"MatGetOrdering_RCM"      ,MatGetOrdering_RCM);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGQMD,      path,"MatGetOrdering_QMD"      ,MatGetOrdering_QMD);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGROWLENGTH,path,"MatGetOrdering_RowLength",MatGetOrdering_RowLength);CHKERRQ(ierr);
#if defined(PETSC_HAVE_UMFPACK)
  ierr = MatOrderingRegisterDynamic(MATORDERINGAMD,      path,"MatGetOrdering_AMD",MatGetOrdering_AMD);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

