#ifndef _SFALLGATHER_H
#define _SFALLGATHER_H

#include <petscsftypes.h>

PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Allgather(PetscSF, PetscInt *, const PetscMPIInt **, const PetscInt **, const PetscInt **);
PETSC_INTERN PetscErrorCode PetscSFSetUp_Allgather(PetscSF);
#endif
