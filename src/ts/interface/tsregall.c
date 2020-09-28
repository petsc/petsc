
#include <petsc/private/tsimpl.h>     /*I  "petscts.h"  I*/
PETSC_EXTERN PetscErrorCode TSCreate_Euler(TS);
PETSC_EXTERN PetscErrorCode TSCreate_BEuler(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Pseudo(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Sundials(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Radau5(TS);
PETSC_EXTERN PetscErrorCode TSCreate_CN(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Theta(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Alpha(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Alpha2(TS);
PETSC_EXTERN PetscErrorCode TSCreate_GLLE(TS);
PETSC_EXTERN PetscErrorCode TSCreate_SSP(TS);
PETSC_EXTERN PetscErrorCode TSCreate_RK(TS);
PETSC_EXTERN PetscErrorCode TSCreate_ARKIMEX(TS);
PETSC_EXTERN PetscErrorCode TSCreate_RosW(TS);
PETSC_EXTERN PetscErrorCode TSCreate_EIMEX(TS);
PETSC_EXTERN PetscErrorCode TSCreate_Mimex(TS);
PETSC_EXTERN PetscErrorCode TSCreate_BDF(TS);
PETSC_EXTERN PetscErrorCode TSCreate_GLEE(TS);
PETSC_EXTERN PetscErrorCode TSCreate_BasicSymplectic(TS);
PETSC_EXTERN PetscErrorCode TSCreate_MPRK(TS);
PETSC_EXTERN PetscErrorCode TSCreate_DiscGrad(TS);

/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso: TSCreate(), TSRegister(), TSRegisterDestroy()
@*/
PetscErrorCode  TSRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSRegisterAllCalled) PetscFunctionReturn(0);
  TSRegisterAllCalled = PETSC_TRUE;

  ierr = TSRegister(TSEULER,          TSCreate_Euler);CHKERRQ(ierr);
  ierr = TSRegister(TSBEULER,         TSCreate_BEuler);CHKERRQ(ierr);
  ierr = TSRegister(TSCN,             TSCreate_CN);CHKERRQ(ierr);
  ierr = TSRegister(TSPSEUDO,         TSCreate_Pseudo);CHKERRQ(ierr);
  ierr = TSRegister(TSGLLE,           TSCreate_GLLE);CHKERRQ(ierr);
  ierr = TSRegister(TSSSP,            TSCreate_SSP);CHKERRQ(ierr);
  ierr = TSRegister(TSTHETA,          TSCreate_Theta);CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA,          TSCreate_Alpha);CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA2,         TSCreate_Alpha2);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUNDIALS)
  ierr = TSRegister(TSSUNDIALS,       TSCreate_Sundials);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_RADAU5)
  ierr = TSRegister(TSRADAU5,         TSCreate_Radau5);CHKERRQ(ierr);
#endif
  ierr = TSRegister(TSRK,             TSCreate_RK);CHKERRQ(ierr);
  ierr = TSRegister(TSGLEE,           TSCreate_GLEE);CHKERRQ(ierr);
  ierr = TSRegister(TSARKIMEX,        TSCreate_ARKIMEX);CHKERRQ(ierr);
  ierr = TSRegister(TSROSW,           TSCreate_RosW);CHKERRQ(ierr);
  ierr = TSRegister(TSEIMEX,          TSCreate_EIMEX);CHKERRQ(ierr);
  ierr = TSRegister(TSMIMEX,          TSCreate_Mimex);CHKERRQ(ierr);
  ierr = TSRegister(TSBDF,            TSCreate_BDF);CHKERRQ(ierr);
  ierr = TSRegister(TSBASICSYMPLECTIC,TSCreate_BasicSymplectic);CHKERRQ(ierr);
  ierr = TSRegister(TSMPRK,           TSCreate_MPRK);CHKERRQ(ierr);
  ierr = TSRegister(TSDISCGRAD,       TSCreate_DiscGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
