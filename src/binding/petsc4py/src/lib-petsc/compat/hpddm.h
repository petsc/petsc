#ifndef PETSC4PY_COMPAT_HPDDM_H
#define PETSC4PY_COMPAT_HPDDM_H

#if !defined(PETSC_HAVE_HPDDM)

#define PetscHPDDMError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires HPDDM",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode KSPHPDDMSetType(PETSC_UNUSED KSP ksp,PETSC_UNUSED KSPHPDDMType type){PetscHPDDMError;}
PetscErrorCode KSPHPDDMGetType(PETSC_UNUSED KSP ksp,PETSC_UNUSED KSPHPDDMType *type){PetscHPDDMError;}

PetscErrorCode PCHPDDMSetAuxiliaryMat(PETSC_UNUSED PC pc,PETSC_UNUSED IS is,PETSC_UNUSED Mat aux,PETSC_UNUSED PetscErrorCode (*setup)(Mat,PetscReal,Vec,Vec,PetscReal,IS,void*),PETSC_UNUSED void* ctx){PetscHPDDMError;}
PetscErrorCode PCHPDDMSetRHSMat(PETSC_UNUSED PC pc,PETSC_UNUSED Mat B){PetscHPDDMError;}
PetscErrorCode PCHPDDMGetComplexities(PETSC_UNUSED PC pc,PETSC_UNUSED PetscReal *gc, PETSC_UNUSED PetscReal *oc){PetscHPDDMError;}
PetscErrorCode PCHPDDMHasNeumannMat(PETSC_UNUSED PC pc,PETSC_UNUSED PetscBool has){PetscHPDDMError;}
PetscErrorCode PCHPDDMSetCoarseCorrectionType(PETSC_UNUSED PC pc,PETSC_UNUSED PCHPDDMCoarseCorrectionType type){PetscHPDDMError;}
PetscErrorCode PCHPDDMGetCoarseCorrectionType(PETSC_UNUSED PC pc,PETSC_UNUSED PCHPDDMCoarseCorrectionType *type){PetscHPDDMError;}
PetscErrorCode PCHPDDMGetSTShareSubKSP(PETSC_UNUSED PC pc,PETSC_UNUSED PetscBool *share){PetscHPDDMError;}
PetscErrorCode PCHPDDMSetDeflationMat(PETSC_UNUSED PC pc,PETSC_UNUSED IS is,PETSC_UNUSED Mat U){PetscHPDDMError;}

#undef PetscHPDDMError

#endif

#endif/*PETSC4PY_COMPAT_HPDDM_H*/
