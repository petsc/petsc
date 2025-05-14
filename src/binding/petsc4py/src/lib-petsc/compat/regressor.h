#ifndef PETSC4PY_COMPAT_REGRESSOR_H
#define PETSC4PY_COMPAT_REGRESSOR_H
#if defined(PETSC_USE_COMPLEX)

#define PetscRegressorError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PetscRegressorLinearSetFitIntercept(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscBool flag) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearSetUseKSP(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscBool flag) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetKSP(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED KSP *ksp) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetCoefficients(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Vec *vec) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetIntercept(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscScalar *intercept) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearSetType(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscRegressorLinearType type) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetType(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscRegressorLinearType *type) {PetscRegressorError;}
#undef PetscRegressorError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_REGRESSOR_H*/
