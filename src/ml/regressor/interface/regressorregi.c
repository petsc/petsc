#include <petsc/private/regressorimpl.h>

PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor);

PetscErrorCode PetscRegressorRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscRegressorRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscRegressorRegisterAllCalled = PETSC_TRUE;
  // Register all of the types of PetscRegressor
#if !PetscDefined(USE_COMPLEX)
  PetscCall(PetscRegressorRegister(PETSCREGRESSORLINEAR, PetscRegressorCreate_Linear));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
