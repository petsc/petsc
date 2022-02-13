#ifndef PETSC4PY_COMPAT_TAO_H
#define PETSC4PY_COMPAT_TAO_H
#if defined(PETSC_USE_COMPLEX)

#define PetscTaoError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode TaoLMVMSetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat mat) {PetscTaoError;}
PetscErrorCode TaoLMVMGetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat *mat) {PetscTaoError;}
PetscErrorCode TaoLMVMGetH0KSP(PETSC_UNUSED Tao tao,PETSC_UNUSED KSP *ksp) {PetscTaoError;}

PetscErrorCode TaoBRGNGetSubsolver(PETSC_UNUSED Tao tao,PETSC_UNUSED Tao *subsolver) {PetscTaoError;}
PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(PETSC_UNUSED Tao tao,PETSC_UNUSED PetscErrorCode (*func)(Tao,Vec,PetscReal*,Vec,void*),PETSC_UNUSED void *ctx) {PetscTaoError;}
PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat H,PETSC_UNUSED PetscErrorCode (*func)(Tao,Vec,Mat,void*),PETSC_UNUSED void *ctx) {PetscTaoError;}
PetscErrorCode TaoBRGNSetRegularizerWeight(PETSC_UNUSED Tao tao,PETSC_UNUSED PetscReal weight) {PetscTaoError;}
PetscErrorCode TaoBRGNSetL1SmoothEpsilon(PETSC_UNUSED Tao tao,PETSC_UNUSED PetscReal epsilon) {PetscTaoError;}
PetscErrorCode TaoBRGNSetDictionaryMatrix(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat D) {PetscTaoError;}
PetscErrorCode TaoBRGNGetDampingVector(PETSC_UNUSED Tao tao,PETSC_UNUSED Vec *d) {PetscTaoError;}

#undef PetscTaoError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_TAO_H*/
