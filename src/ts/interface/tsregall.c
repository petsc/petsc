
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

.seealso: `TSCreate()`, `TSRegister()`, `TSRegisterDestroy()`
@*/
PetscErrorCode  TSRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSRegisterAllCalled) PetscFunctionReturn(0);
  TSRegisterAllCalled = PETSC_TRUE;

  PetscCall(TSRegister(TSEULER,          TSCreate_Euler));
  PetscCall(TSRegister(TSBEULER,         TSCreate_BEuler));
  PetscCall(TSRegister(TSCN,             TSCreate_CN));
  PetscCall(TSRegister(TSPSEUDO,         TSCreate_Pseudo));
  PetscCall(TSRegister(TSGLLE,           TSCreate_GLLE));
  PetscCall(TSRegister(TSSSP,            TSCreate_SSP));
  PetscCall(TSRegister(TSTHETA,          TSCreate_Theta));
  PetscCall(TSRegister(TSALPHA,          TSCreate_Alpha));
  PetscCall(TSRegister(TSALPHA2,         TSCreate_Alpha2));
#if defined(PETSC_HAVE_SUNDIALS2)
  PetscCall(TSRegister(TSSUNDIALS,       TSCreate_Sundials));
#endif
#if defined(PETSC_HAVE_RADAU5)
  PetscCall(TSRegister(TSRADAU5,         TSCreate_Radau5));
#endif
  PetscCall(TSRegister(TSRK,             TSCreate_RK));
  PetscCall(TSRegister(TSGLEE,           TSCreate_GLEE));
  PetscCall(TSRegister(TSARKIMEX,        TSCreate_ARKIMEX));
  PetscCall(TSRegister(TSROSW,           TSCreate_RosW));
  PetscCall(TSRegister(TSEIMEX,          TSCreate_EIMEX));
  PetscCall(TSRegister(TSMIMEX,          TSCreate_Mimex));
  PetscCall(TSRegister(TSBDF,            TSCreate_BDF));
  PetscCall(TSRegister(TSBASICSYMPLECTIC,TSCreate_BasicSymplectic));
  PetscCall(TSRegister(TSMPRK,           TSCreate_MPRK));
  PetscCall(TSRegister(TSDISCGRAD,       TSCreate_DiscGrad));
  PetscCall(TSRegister(TSIRK,            TSCreate_IRK));
  PetscFunctionReturn(0);
}
