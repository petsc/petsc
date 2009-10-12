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
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_SSP(TS);
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

  ierr = TSRegisterDynamic(TSEULER,           path, "TSCreate_Euler",    TSCreate_Euler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSBEULER,          path, "TSCreate_BEuler",   TSCreate_BEuler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSCRANK_NICHOLSON, path, "TSCreate_CN",       TSCreate_CN);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSPSEUDO,          path, "TSCreate_Pseudo",   TSCreate_Pseudo);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSGL,              path, "TSCreate_GL",       TSCreate_GL);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSSSP,             path, "TSCreate_SSP",      TSCreate_SSP);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSTHETA,           path, "TSCreate_Theta",    TSCreate_Theta);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUNDIALS)
  ierr = TSRegisterDynamic(TSSUNDIALS,        path, "TSCreate_Sundials", TSCreate_Sundials);CHKERRQ(ierr);
#endif
  ierr = TSRegisterDynamic(TSRUNGE_KUTTA,     path, "TSCreate_Rk",       TSCreate_Rk);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

