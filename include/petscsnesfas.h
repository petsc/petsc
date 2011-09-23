#if !(defined __PETSCSNESFAS_H)
#define __PETSCSNESFAS_H
#include "petscsnes.h"

extern PetscErrorCode SNESFASSetLevels(SNES, PetscInt, MPI_Comm *);
extern PetscErrorCode SNESFASGetLevels(SNES, PetscInt *);
extern PetscErrorCode SNESFASGetSNES(SNES, PetscInt, SNES *);
extern PetscErrorCode SNESFASGetPreSmoother(SNES, PetscInt, SNES *);
extern PetscErrorCode SNESFASGetPostSmoother(SNES, PetscInt, SNES *);

extern PetscErrorCode SNESFASSetInterpolation(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetRestriction(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetRScale(SNES, PetscInt, Vec);

#endif
