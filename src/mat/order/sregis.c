#define PETSCMAT_DLL

#include "private/matimpl.h"     /*I       "petscmat.h"   I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_Natural(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_ND(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_1WD(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_QMD(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_RCM(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_RowLength(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_DSC(Mat,const MatOrderingType,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_Flow_SeqAIJ(Mat,const MatOrderingType,IS*,IS*);
#if defined(PETSC_HAVE_UMFPACK)
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_AMD(Mat,const MatOrderingType,IS*,IS*);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatOrderingRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatOrderingRegisterAllCalled = PETSC_TRUE;

  ierr = MatOrderingRegisterDynamic(MATORDERINGNATURAL,  path,"MatOrdering_Natural"  ,MatOrdering_Natural);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGND,       path,"MatOrdering_ND"       ,MatOrdering_ND);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERING1WD,      path,"MatOrdering_1WD"      ,MatOrdering_1WD);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGRCM,      path,"MatOrdering_RCM"      ,MatOrdering_RCM);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGQMD,      path,"MatOrdering_QMD"      ,MatOrdering_QMD);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGROWLENGTH,path,"MatOrdering_RowLength",MatOrdering_RowLength);CHKERRQ(ierr);
  ierr = MatOrderingRegisterDynamic(MATORDERINGFLOW,     path,"MatOrdering_Flow_SeqAIJ",MatOrdering_Flow_SeqAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_UMFPACK)
  ierr = MatOrderingRegisterDynamic(MATORDERINGAMD,      path,"MatOrdering_AMD",MatOrdering_AMD);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

