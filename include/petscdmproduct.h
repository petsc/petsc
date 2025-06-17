#pragma once

#include <petscdm.h>

/* MANSEC = DM */

PETSC_EXTERN PetscErrorCode DMCreate_Product(DM);
PETSC_EXTERN PetscErrorCode DMProductGetDM(DM, PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMProductSetDimensionIndex(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMProductGetDimensionIndex(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMProductSetDM(DM, PetscInt, DM);
