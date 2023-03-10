#ifndef PETSC4PY_COMPAT_MUMPS_H
#define PETSC4PY_COMPAT_MUMPS_H

#include <petscmat.h>
#if !defined(PETSC_HAVE_MUMPS)

#define PetscMUMPSError do {                    \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires MUMPS",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode MatMumpsSetIcntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt ival){PetscMUMPSError;}
PetscErrorCode MatMumpsGetIcntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt *ival){PetscMUMPSError;}
PetscErrorCode MatMumpsSetCntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal val){PetscMUMPSError;}
PetscErrorCode MatMumpsGetCntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal *val){PetscMUMPSError;}
PetscErrorCode MatMumpsGetInfo(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt *ival){PetscMUMPSError;}
PetscErrorCode MatMumpsGetInfog(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt *ival){PetscMUMPSError;}
PetscErrorCode MatMumpsGetRinfo(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal *val){PetscMUMPSError;}
PetscErrorCode MatMumpsGetRinfog(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal *val){PetscMUMPSError;}

#undef PetscMUMPSError

#endif

#endif/*PETSC4PY_COMPAT_MUMPS_H*/
