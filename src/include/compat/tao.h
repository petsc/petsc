#ifndef PETSC4PY_COMPAT_TAO_H
#define PETSC4PY_COMPAT_TAO_H
#if defined(PETSC_USE_COMPLEX)

#define PetscTaoError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported with complex scalars"); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

#undef __FUNCT__
#define __FUNCT__ "TaoLMVMSetH0"
PetscErrorCode TaoLMVMSetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat mat) {PetscTaoError;}

#undef __FUNCT__
#define __FUNCT__ "TaoLMVMGetH0"
PetscErrorCode TaoLMVMGetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat *mat) {PetscTaoError;}

#undef __FUNCT__
#define __FUNCT__ "TaoLMVMGetH0KSP"
PetscErrorCode TaoLMVMGetH0KSP(PETSC_UNUSED Tao tao,PETSC_UNUSED KSP *ksp) {PetscTaoError;}

#undef PetscTaoError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_TAO_H*/
