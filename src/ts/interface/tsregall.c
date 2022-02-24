
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
PETSC_EXTERN PetscErrorCode TSCreate_IRK(TS);

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
  PetscFunctionBegin;
  if (TSRegisterAllCalled) PetscFunctionReturn(0);
  TSRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(TSRegister(TSEULER,          TSCreate_Euler));
  CHKERRQ(TSRegister(TSBEULER,         TSCreate_BEuler));
  CHKERRQ(TSRegister(TSCN,             TSCreate_CN));
  CHKERRQ(TSRegister(TSPSEUDO,         TSCreate_Pseudo));
  CHKERRQ(TSRegister(TSGLLE,           TSCreate_GLLE));
  CHKERRQ(TSRegister(TSSSP,            TSCreate_SSP));
  CHKERRQ(TSRegister(TSTHETA,          TSCreate_Theta));
  CHKERRQ(TSRegister(TSALPHA,          TSCreate_Alpha));
  CHKERRQ(TSRegister(TSALPHA2,         TSCreate_Alpha2));
#if defined(PETSC_HAVE_SUNDIALS2)
  CHKERRQ(TSRegister(TSSUNDIALS,       TSCreate_Sundials));
#endif
#if defined(PETSC_HAVE_RADAU5)
  CHKERRQ(TSRegister(TSRADAU5,         TSCreate_Radau5));
#endif
  CHKERRQ(TSRegister(TSRK,             TSCreate_RK));
  CHKERRQ(TSRegister(TSGLEE,           TSCreate_GLEE));
  CHKERRQ(TSRegister(TSARKIMEX,        TSCreate_ARKIMEX));
  CHKERRQ(TSRegister(TSROSW,           TSCreate_RosW));
  CHKERRQ(TSRegister(TSEIMEX,          TSCreate_EIMEX));
  CHKERRQ(TSRegister(TSMIMEX,          TSCreate_Mimex));
  CHKERRQ(TSRegister(TSBDF,            TSCreate_BDF));
  CHKERRQ(TSRegister(TSBASICSYMPLECTIC,TSCreate_BasicSymplectic));
  CHKERRQ(TSRegister(TSMPRK,           TSCreate_MPRK));
  CHKERRQ(TSRegister(TSDISCGRAD,       TSCreate_DiscGrad));
  CHKERRQ(TSRegister(TSIRK,            TSCreate_IRK));
  PetscFunctionReturn(0);
}
