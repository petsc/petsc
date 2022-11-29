#ifndef PETSC_DMPRODUCT_H
#define PETSC_DMPRODUCT_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMCreate_Product(DM);
PETSC_EXTERN PetscErrorCode DMProductGetDM(DM, PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMProductSetDimensionIndex(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMProductSetDM(DM, PetscInt, DM);

#endif // PETSC_DMPRODUCT_H
