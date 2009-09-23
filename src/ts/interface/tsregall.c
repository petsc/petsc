#define PETSCTS_DLL

#include "private/tsimpl.h"     /*I  "petscts.h"  I*/
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Euler(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_BEuler(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Pseudo(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Sundials(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_CN(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Theta(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_GL(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Rk(TS);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "TSRegisterAll"
/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS package. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: TS, timestepper, register, all
.seealso: TSCreate(), TSRegister(), TSRegisterDestroy(), TSRegisterDynamic()
@*/
PetscErrorCode PETSCTS_DLLEXPORT TSRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRegisterAllCalled = PETSC_TRUE;

  ierr = TSRegisterDynamic(TS_EULER,           path, "TSCreate_Euler", TSCreate_Euler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_BEULER,          path, "TSCreate_BEuler",TSCreate_BEuler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_CRANK_NICHOLSON, path, "TSCreate_CN",    TSCreate_CN);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_PSEUDO,          path, "TSCreate_Pseudo",TSCreate_Pseudo);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_GENERAL_LINEAR,  path, "TSCreate_GL",    TSCreate_GL);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_THETA,           path, "TSCreate_Theta", TSCreate_Theta);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUNDIALS)
  ierr = TSRegisterDynamic(TS_SUNDIALS,           path, "TSCreate_Sundials", TSCreate_Sundials);CHKERRQ(ierr);
#endif
  ierr = TSRegisterDynamic(TS_RUNGE_KUTTA,     path, "TSCreate_Rk", TSCreate_Rk);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

