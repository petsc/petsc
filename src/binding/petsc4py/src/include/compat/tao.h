#ifndef PETSC4PY_COMPAT_TAO_H
#define PETSC4PY_COMPAT_TAO_H
#if defined(PETSC_USE_COMPLEX)

#define PetscTaoError do { \
    PetscFunctionBegin; \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode TaoLMVMSetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat mat) {PetscTaoError;}
PetscErrorCode TaoLMVMGetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat *mat) {PetscTaoError;}
PetscErrorCode TaoLMVMGetH0KSP(PETSC_UNUSED Tao tao,PETSC_UNUSED KSP *ksp) {PetscTaoError;}

#undef PetscTaoError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_TAO_H*/
