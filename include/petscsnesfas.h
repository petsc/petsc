#if !(defined __PETSCSNESFAS_H)
#define __PETSCSNESFAS_H
#include "petscsnes.h"


/*E
    SNESFASType - Determines the type of multigrid method that is run.

   Level: beginner

   Values:
+  SNES_FAS_MULTIPLICATIVE (default) - traditional V or W cycle as determined by SNESFASSetCycles()
-  SNES_FAS_ADDITIVE - the additive multigrid preconditioner where all levels are

.seealso: PCMGSetType(), PCMGType

E*/
typedef enum { SNES_FAS_MULTIPLICATIVE, SNES_FAS_ADDITIVE } SNESFASType;
PETSC_EXTERN const char * SNESFASTypes[];

/* called on the finest level FAS instance*/
PETSC_EXTERN PetscErrorCode SNESFASSetType(SNES, SNESFASType);
PETSC_EXTERN PetscErrorCode SNESFASGetType(SNES, SNESFASType*);
PETSC_EXTERN PetscErrorCode SNESFASSetLevels(SNES, PetscInt, MPI_Comm *);
PETSC_EXTERN PetscErrorCode SNESFASGetLevels(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESFASGetCycleSNES(SNES, PetscInt, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASSetNumberSmoothUp(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetNumberSmoothDown(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetCycles(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetMonitor(SNES, PetscBool);


PETSC_EXTERN PetscErrorCode SNESFASSetGalerkin(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASGetGalerkin(SNES, PetscBool*);

/* called on any level -- "Cycle" FAS instance */
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmoother(SNES, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmootherUp(SNES, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmootherDown(SNES, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetCorrection(SNES, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetInterpolation(SNES, Mat*);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetRestriction(SNES, Mat*);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetInjection(SNES, Mat*);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetRScale(SNES, Vec*);
PETSC_EXTERN PetscErrorCode SNESFASCycleSetCycles(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASCycleIsFine(SNES, PetscBool*);

/* called on the (outer) finest level FAS to set/get parameters on any level instance */
PETSC_EXTERN PetscErrorCode SNESFASSetInterpolation(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetInterpolation(SNES, PetscInt, Mat*);
PETSC_EXTERN PetscErrorCode SNESFASSetRestriction(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetRestriction(SNES, PetscInt, Mat*);
PETSC_EXTERN PetscErrorCode SNESFASSetInjection(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetInjection(SNES, PetscInt, Mat*);
PETSC_EXTERN PetscErrorCode SNESFASSetRScale(SNES, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode SNESFASGetRScale(SNES, PetscInt, Vec*);

PETSC_EXTERN PetscErrorCode SNESFASGetSmoother(SNES,     PetscInt, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASGetSmootherUp(SNES,   PetscInt, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASGetSmootherDown(SNES, PetscInt, SNES*);
PETSC_EXTERN PetscErrorCode SNESFASGetCoarseSolve(SNES, SNES*);

PETSC_EXTERN PetscErrorCode SNESFASCreateCoarseVec(SNES,Vec*);
PETSC_EXTERN PetscErrorCode SNESFASRestrict(SNES,Vec,Vec);

#endif
