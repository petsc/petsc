#ifndef PETSC4PY_COMPAT_HPDDM_H
#define PETSC4PY_COMPAT_HPDDM_H

#if !defined(PETSC_HAVE_HPDDM) || !defined(PETSC_HAVE_DYNAMIC_LIBRARIES) || !defined(PETSC_USE_SHARED_LIBRARIES)

#define PetscPCHPDDMError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires HPDDM",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PCHPDDMSetAuxiliaryMat(PETSC_UNUSED PC pc,PETSC_UNUSED IS is,PETSC_UNUSED Mat aux,PETSC_UNUSED PetscErrorCode (*setup)(Mat,PetscReal,Vec,Vec,PetscReal,IS,void*),PETSC_UNUSED void* ctx){PetscPCHPDDMError;}
PetscErrorCode PCHPDDMSetRHSMat(PETSC_UNUSED PC pc,PETSC_UNUSED Mat B){PetscPCHPDDMError;}
PetscErrorCode PCHPDDMHasNeumannMat(PETSC_UNUSED PC pc,PETSC_UNUSED PetscBool has){PetscPCHPDDMError;}
PetscErrorCode PCHPDDMSetCoarseCorrectionType(PETSC_UNUSED PC pc,PETSC_UNUSED PCHPDDMCoarseCorrectionType type){PetscPCHPDDMError;}
PetscErrorCode PCHPDDMGetCoarseCorrectionType(PETSC_UNUSED PC pc,PETSC_UNUSED PCHPDDMCoarseCorrectionType *type){PetscPCHPDDMError;}
PetscErrorCode PCHPDDMGetSTShareSubKSP(PETSC_UNUSED PC pc,PETSC_UNUSED PetscBool *share){PetscPCHPDDMError;}

#undef PetscPCHPDDMError

#endif

#endif/*PETSC4PY_COMPAT_HPDDM_H*/
