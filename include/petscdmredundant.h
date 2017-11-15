/* DM for redundant globally coupled degrees of freedom */
#if !defined(__PETSCDMREDUNDANT_H)
#define __PETSCDMREDUNDANT_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMRedundantCreate(MPI_Comm,PetscInt,PetscInt,DM*);
PETSC_EXTERN PetscErrorCode DMRedundantSetSize(DM,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMRedundantGetSize(DM,PetscInt*,PetscInt*);

#endif
