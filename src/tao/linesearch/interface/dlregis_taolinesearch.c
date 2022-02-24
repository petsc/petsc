#define TAOLINESEARCH_DLL
#include <petsc/private/taolinesearchimpl.h>

PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_Unit(TaoLineSearch);
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_MT(TaoLineSearch);
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_GPCG(TaoLineSearch);
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_Armijo(TaoLineSearch);
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_OWArmijo(TaoLineSearch);
static PetscBool TaoLineSearchPackageInitialized = PETSC_FALSE;

/*@C
  TaoLineSearchFinalizePackage - This function destroys everything in the PETSc/TAO
  interface to the TaoLineSearch package. It is called from PetscFinalize().

  Level: developer
@*/
PetscErrorCode TaoLineSearchFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&TaoLineSearchList));
  TaoLineSearchPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TaoLineSearchInitializePackage - This function registers the line-search
  algorithms in TAO.  When using shared or static libraries, this function is called from the
  first entry to TaoCreate(); when using dynamic, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

.seealso: TaoLineSearchCreate()
@*/
PetscErrorCode TaoLineSearchInitializePackage(void)
{
  PetscFunctionBegin;
  if (TaoLineSearchPackageInitialized) PetscFunctionReturn(0);
  TaoLineSearchPackageInitialized=PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscClassIdRegister("TaoLineSearch",&TAOLINESEARCH_CLASSID));
  CHKERRQ(TaoLineSearchRegister("unit",TaoLineSearchCreate_Unit));
  CHKERRQ(TaoLineSearchRegister("more-thuente",TaoLineSearchCreate_MT));
  CHKERRQ(TaoLineSearchRegister("gpcg",TaoLineSearchCreate_GPCG));
  CHKERRQ(TaoLineSearchRegister("armijo",TaoLineSearchCreate_Armijo));
  CHKERRQ(TaoLineSearchRegister("owarmijo",TaoLineSearchCreate_OWArmijo));
  CHKERRQ(PetscLogEventRegister("TaoLSApply",TAOLINESEARCH_CLASSID,&TAOLINESEARCH_Apply));
  CHKERRQ(PetscLogEventRegister("TaoLSEval", TAOLINESEARCH_CLASSID,&TAOLINESEARCH_Eval));
#endif
  CHKERRQ(PetscRegisterFinalize(TaoLineSearchFinalizePackage));
  PetscFunctionReturn(0);
}
