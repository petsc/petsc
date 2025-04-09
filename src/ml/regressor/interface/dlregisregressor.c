#include <petsc/private/regressorimpl.h>

static PetscBool PetscRegressorPackageInitialized = PETSC_FALSE;

/*@C
  PetscRegressorInitializePackage - Initialize `PetscRegressor` package

  Logically Collective

  Level: developer

.seealso: `PetscRegressorFinalizePackage()`
@*/
PetscErrorCode PetscRegressorInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscRegressorPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscRegressorPackageInitialized = PETSC_TRUE;
  /* Register Class */
  PetscCall(PetscClassIdRegister("Regressor", &PETSCREGRESSOR_CLASSID));
  /* Register Constructors */
  PetscCall(PetscRegressorRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("PetscRegressorSetUp", PETSCREGRESSOR_CLASSID, &PetscRegressor_SetUp));
  PetscCall(PetscLogEventRegister("PetscRegressorFit", PETSCREGRESSOR_CLASSID, &PetscRegressor_Fit));
  PetscCall(PetscLogEventRegister("PetscRegressorPredict", PETSCREGRESSOR_CLASSID, &PetscRegressor_Predict));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PETSCREGRESSOR_CLASSID;
    PetscCall(PetscInfoProcessClass("petscregressor", 1, classids));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscRegressorFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRegressorFinalizePackage - Finalize `PetscRegressor` package; it is called from `PetscFinalize()`

  Logically Collective

  Level: developer

.seealso: `PetscRegressorInitializePackage()`
@*/
PetscErrorCode PetscRegressorFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscRegressorList));
  PetscRegressorPackageInitialized = PETSC_FALSE;
  PetscRegressorRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
