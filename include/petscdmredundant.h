/* DM for redundant globally coupled degrees of freedom */
#pragma once

#include <petscdm.h>

/* MANSEC = DM */

PETSC_EXTERN PetscErrorCode DMRedundantCreate(MPI_Comm, PetscMPIInt, PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMRedundantSetSize(DM, PetscMPIInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMRedundantGetSize(DM, PetscMPIInt *, PetscInt *);
