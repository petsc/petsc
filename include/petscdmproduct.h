#if !defined(DMPRODUCT_H_)
#define DMPRODUCT_H_

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMCreate_Product(DM);
PETSC_EXTERN PetscErrorCode DMProductGetDM(DM,PetscInt,DM*);
PETSC_EXTERN PetscErrorCode DMProductSetDimensionIndex(DM,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMProductSetDM(DM,PetscInt,DM);

#endif
