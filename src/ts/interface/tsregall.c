
#include <petsc-private/tsimpl.h>     /*I  "petscts.h"  I*/
PETSC_EXTERN PetscErrorCode TSCreate_Euler(TS);
PETSC_EXTERN PetscErrorCode TSCreate_BEuler(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Pseudo(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Sundials(TS);
PETSC_EXTERN PetscErrorCode TSCreate_CN(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Theta(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Alpha(TS);
PETSC_EXTERN PetscErrorCode TSCreate_GL(TS);
PETSC_EXTERN PetscErrorCode TSCreate_SSP(TS);
PETSC_EXTERN PetscErrorCode TSCreate_RK(TS);
PETSC_EXTERN PetscErrorCode TSCreate_ARKIMEX(TS);
PETSC_EXTERN PetscErrorCode TSCreate_RosW(TS);

#undef __FUNCT__
#define __FUNCT__ "TSRegisterAll"
/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: TS, timestepper, register, all
.seealso: TSCreate(), TSRegister(), TSRegisterDestroy(), TSRegister()
@*/
PetscErrorCode  TSRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRegisterAllCalled = PETSC_TRUE;

  ierr = TSRegister(TSEULER,           path, "TSCreate_Euler",    TSCreate_Euler);CHKERRQ(ierr);
  ierr = TSRegister(TSBEULER,          path, "TSCreate_BEuler",   TSCreate_BEuler);CHKERRQ(ierr);
  ierr = TSRegister(TSCN,              path, "TSCreate_CN",       TSCreate_CN);CHKERRQ(ierr);
  ierr = TSRegister(TSPSEUDO,          path, "TSCreate_Pseudo",   TSCreate_Pseudo);CHKERRQ(ierr);
  ierr = TSRegister(TSGL,              path, "TSCreate_GL",       TSCreate_GL);CHKERRQ(ierr);
  ierr = TSRegister(TSSSP,             path, "TSCreate_SSP",      TSCreate_SSP);CHKERRQ(ierr);
  ierr = TSRegister(TSTHETA,           path, "TSCreate_Theta",    TSCreate_Theta);CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA,           path, "TSCreate_Alpha",    TSCreate_Alpha);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUNDIALS)
  ierr = TSRegister(TSSUNDIALS,        path, "TSCreate_Sundials", TSCreate_Sundials);CHKERRQ(ierr);
#endif
  ierr = TSRegister(TSRK,              path, "TSCreate_RK",       TSCreate_RK);CHKERRQ(ierr);
  ierr = TSRegister(TSARKIMEX,         path, "TSCreate_ARKIMEX",  TSCreate_ARKIMEX);CHKERRQ(ierr);
  ierr = TSRegister(TSROSW,            path, "TSCreate_RosW",     TSCreate_RosW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

