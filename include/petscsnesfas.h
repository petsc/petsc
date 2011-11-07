#if !(defined __PETSCSNESFAS_H)
#define __PETSCSNESFAS_H
#include "petscsnes.h"

extern PetscErrorCode SNESFASSetLevels(SNES, PetscInt, MPI_Comm *);
extern PetscErrorCode SNESFASGetLevels(SNES, PetscInt *);

extern PetscErrorCode SNESFASSetCycles(SNES, PetscInt);
extern PetscErrorCode SNESFASSetCyclesOnLevel(SNES, PetscInt, PetscInt);

extern PetscErrorCode SNESFASGetSNES(SNES, PetscInt, SNES *);
extern PetscErrorCode SNESFASSetNumberSmoothUp(SNES, PetscInt);
extern PetscErrorCode SNESFASSetNumberSmoothDown(SNES, PetscInt);

extern PetscErrorCode SNESFASGetSmoother(SNES, PetscInt, SNES *);
extern PetscErrorCode SNESFASGetSmootherUp(SNES, PetscInt, SNES *);
extern PetscErrorCode SNESFASGetSmootherDown(SNES, PetscInt, SNES *);
extern PetscErrorCode SNESFASGetCoarseSolve(SNES, SNES *);

extern PetscErrorCode SNESFASSetInterpolation(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetRestriction(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetInjection(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetRScale(SNES, PetscInt, Vec);

#endif
