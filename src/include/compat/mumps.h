#ifndef PETSC4PY_COMPAT_MUMPS_H
#define PETSC4PY_COMPAT_MUMPS_H

#include <petscmat.h>
#if !defined(PETSC_HAVE_MUMPS)
#define PetscMUMPSError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() requires MUMPS"); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)
#undef __FUNCT__
#define __FUNCT__ "MatMumpsSetIcntl"
PetscErrorCode MatMumpsSetIcntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt ival){PetscMUMPSError;}
#undef __FUNCT__
#define __FUNCT__ "MatMumpsGetIcntl"
PetscErrorCode MatMumpsGetIcntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt *ival){PetscMUMPSError;}
#undef __FUNCT__
#define __FUNCT__ "MatMumpsSetCntl"
PetscErrorCode MatMumpsSetCntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal val){PetscMUMPSError;}
#undef __FUNCT__
#define __FUNCT__ "MatMumpsGetCntl"
PetscErrorCode MatMumpsGetCntl(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal *val){PetscMUMPSError;}
#undef __FUNCT__
#define __FUNCT__ "MatMumpsGetInfo"
PetscErrorCode MatMumpsGetInfo(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt *ival){PetscMUMPSError;}
#undef __FUNCT__
#define __FUNCT__ "MatMumpsGetInfog"
PetscErrorCode MatMumpsGetInfog(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscInt *ival){PetscMUMPSError;}
#undef __FUNCT__
#define __FUNCT__ "MatMumpsGetRinfo"
PetscErrorCode MatMumpsGetRinfo(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal *val){PetscMUMPSError;}
#undef __FUNCT__
#define __FUNCT__ "MatMumpsGetRinfog"
PetscErrorCode MatMumpsGetRinfog(PETSC_UNUSED Mat F,PETSC_UNUSED PetscInt icntl,PETSC_UNUSED PetscReal *val){PetscMUMPSError;}
#undef PetscMUMPSError
#endif

#endif/*PETSC4PY_COMPAT_MUMPS_H*/
