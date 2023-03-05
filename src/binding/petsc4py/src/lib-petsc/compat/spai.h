#ifndef PETSC4PY_COMPAT_SPAI_H
#define PETSC4PY_COMPAT_SPAI_H

#if !defined(PETSC_HAVE_SPAI)

#define PetscSPAIError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires SPAI",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PCSPAISetEpsilon(PETSC_UNUSED PC a,PETSC_UNUSED PetscReal b) {PetscSPAIError;}
PetscErrorCode PCSPAISetNBSteps(PETSC_UNUSED PC a,PETSC_UNUSED PetscInt b) {PetscSPAIError;}
PetscErrorCode PCSPAISetMax(PETSC_UNUSED PC a,PETSC_UNUSED PetscInt b) {PetscSPAIError;}
PetscErrorCode PCSPAISetMaxNew(PETSC_UNUSED PC a,PETSC_UNUSED PetscInt b) {PetscSPAIError;}
PetscErrorCode PCSPAISetBlockSize(PETSC_UNUSED PC a,PETSC_UNUSED PetscInt b) {PetscSPAIError;}
PetscErrorCode PCSPAISetCacheSize(PETSC_UNUSED PC a,PETSC_UNUSED PetscInt b) {PetscSPAIError;}
PetscErrorCode PCSPAISetVerbose(PETSC_UNUSED PC a,PETSC_UNUSED PetscInt b) {PetscSPAIError;}
PetscErrorCode PCSPAISetSp(PETSC_UNUSED PC a,PETSC_UNUSED PetscInt b) {PetscSPAIError;}

#undef PetscSPAIError

#endif

#endif/*PETSC4PY_COMPAT_SPAI_H*/
