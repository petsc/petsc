
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
PETSC_EXTERN PetscErrorCode TSCreate_EIMEX(TS);
PETSC_EXTERN PetscErrorCode TSCreate_DAESimple_Reduced(TS);
PETSC_EXTERN PetscErrorCode TSCreate_DAESimple_Full(TS);

#undef __FUNCT__
#define __FUNCT__ "TSRegisterAll"
/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: TS, timestepper, register, all
.seealso: TSCreate(), TSRegister(), TSRegisterDestroy()
@*/
PetscErrorCode  TSRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRegisterAllCalled = PETSC_TRUE;

  ierr = TSRegister(TSEULER,    TSCreate_Euler);CHKERRQ(ierr);
  ierr = TSRegister(TSBEULER,   TSCreate_BEuler);CHKERRQ(ierr);
  ierr = TSRegister(TSCN,       TSCreate_CN);CHKERRQ(ierr);
  ierr = TSRegister(TSPSEUDO,   TSCreate_Pseudo);CHKERRQ(ierr);
  ierr = TSRegister(TSGL,       TSCreate_GL);CHKERRQ(ierr);
  ierr = TSRegister(TSSSP,      TSCreate_SSP);CHKERRQ(ierr);
  ierr = TSRegister(TSTHETA,    TSCreate_Theta);CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA,    TSCreate_Alpha);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUNDIALS)
  ierr = TSRegister(TSSUNDIALS, TSCreate_Sundials);CHKERRQ(ierr);
#endif
  ierr = TSRegister(TSRK,       TSCreate_RK);CHKERRQ(ierr);
  ierr = TSRegister(TSARKIMEX,  TSCreate_ARKIMEX);CHKERRQ(ierr);
  ierr = TSRegister(TSROSW,     TSCreate_RosW);CHKERRQ(ierr);
  ierr = TSRegister(TSEIMEX,    TSCreate_EIMEX);CHKERRQ(ierr);
  ierr = TSRegister(TSDAESIMPLERED, TSCreate_DAESimple_Reduced);CHKERRQ(ierr);
  ierr = TSRegister(TSDAESIMPLEFULL, TSCreate_DAESimple_Full);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

