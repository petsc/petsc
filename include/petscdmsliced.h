/* Very minimal unstructured DM */
#pragma once

#include <petscdm.h>

/* MANSEC = DM */

PETSC_EXTERN PetscErrorCode DMSlicedCreate(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscInt[], DM *);
PETSC_EXTERN PetscErrorCode DMSlicedSetPreallocation(DM, PetscInt, const PetscInt[], PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMSlicedSetBlockFills(DM, const PetscInt *, const PetscInt *);
PETSC_EXTERN PetscErrorCode DMSlicedSetGhosts(DM, PetscInt, PetscInt, PetscInt, const PetscInt[]);
