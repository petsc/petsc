#pragma once

#include <petscdevice.h>

PETSC_INTERN PetscErrorCode AXPBYCyclic_CUPM_Private(PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscScalar, PetscScalar[], PetscInt);
PETSC_INTERN PetscErrorCode DMVCyclic_CUPM_Private(PetscBool, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], const PetscScalar[], PetscScalar, PetscScalar[]);
PETSC_INTERN PetscErrorCode DSVCyclic_CUPM_Private(PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode TRSVCyclic_CUPM_Private(PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[]);
PETSC_INTERN PetscErrorCode GEMVCyclic_CUPM_Private(PetscBool, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar, PetscScalar[]);
PETSC_INTERN PetscErrorCode HEMVCyclic_CUPM_Private(PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar, PetscScalar[]);
