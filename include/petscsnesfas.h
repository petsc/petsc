#if !(defined __PETSCSNESFAS_H)
#define __PETSCSNESFAS_H
#include "petscsnes.h"

typedef enum { SNES_FAS_MULTIPLICATIVE, SNES_FAS_ADDITIVE } SNESFASType;

extern PetscErrorCode SNESFASSetLevels(SNES, PetscInt, MPI_Comm *);
extern PetscErrorCode SNESFASGetLevels(SNES, PetscInt *);

extern PetscErrorCode SNESFASSetCycles(SNES, PetscInt);
extern PetscErrorCode SNESFASSetCyclesOnLevel(SNES, PetscInt, PetscInt);

extern PetscErrorCode SNESFASGetSNES(SNES, PetscInt, SNES *);
extern PetscErrorCode SNESFASSetNumberSmoothUp(SNES, PetscInt);
extern PetscErrorCode SNESFASSetNumberSmoothDown(SNES, PetscInt);

extern PetscErrorCode SNESFASSetInterpolation(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetRestriction(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetInjection(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASSetRScale(SNES, PetscInt, Vec);

extern PetscErrorCode SNESFASSetGS(SNES, PetscErrorCode (*)(SNES,Vec,Vec,void *), void *, PetscBool);
extern PetscErrorCode SNESFASSetGSOnLevel(SNES, PetscInt, PetscErrorCode (*)(SNES,Vec,Vec,void *), void *, PetscBool);

#endif
