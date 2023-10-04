#pragma once

#include <petscsftypes.h>

PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Allgather(PetscSF, PetscInt *, const PetscMPIInt **, const PetscInt **, const PetscInt **);
PETSC_INTERN PetscErrorCode PetscSFSetUp_Allgather(PetscSF);
