#ifndef PETSC4PY_COMPAT_H2OPUS_H
#define PETSC4PY_COMPAT_H2OPUS_H

#if !defined(PETSC_HAVE_H2OPUS)

#define PetscMatH2OPUSError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires H2OPUS",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode MatCreateH2OpusFromMat(PETSC_UNUSED Mat a,PETSC_UNUSED PetscInt b,PETSC_UNUSED const PetscReal c[],PETSC_UNUSED PetscBool d,PETSC_UNUSED PetscReal e,PETSC_UNUSED PetscInt f,PETSC_UNUSED PetscInt g,PETSC_UNUSED PetscInt h,PETSC_UNUSED PetscReal i,PETSC_UNUSED Mat* l) {PetscMatH2OPUSError;}

#undef PetscMatH2OPUSError

#endif

#endif/*PETSC4PY_COMPAT_H2OPUS_H*/
