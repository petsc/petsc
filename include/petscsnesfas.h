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
extern const char * SNESFASTypes[];

/* called on the finest level FAS instance*/
extern PetscErrorCode SNESFASSetType(SNES, SNESFASType);
extern PetscErrorCode SNESFASGetType(SNES, SNESFASType*);
extern PetscErrorCode SNESFASSetLevels(SNES, PetscInt, MPI_Comm *);
extern PetscErrorCode SNESFASGetLevels(SNES, PetscInt *);
extern PetscErrorCode SNESFASGetCycleSNES(SNES, PetscInt, SNES*);
extern PetscErrorCode SNESFASSetNumberSmoothUp(SNES, PetscInt);
extern PetscErrorCode SNESFASSetNumberSmoothDown(SNES, PetscInt);
extern PetscErrorCode SNESFASSetCycles(SNES, PetscInt);
extern PetscErrorCode SNESFASSetMonitor(SNES, PetscBool);


extern PetscErrorCode SNESFASSetGalerkin(SNES, PetscBool);
extern PetscErrorCode SNESFASGetGalerkin(SNES, PetscBool*);

/* called on any level -- "Cycle" FAS instance */
extern PetscErrorCode SNESFASCycleGetSmoother(SNES, SNES*);
extern PetscErrorCode SNESFASCycleGetSmootherUp(SNES, SNES*);
extern PetscErrorCode SNESFASCycleGetSmootherDown(SNES, SNES*);
extern PetscErrorCode SNESFASCycleGetCorrection(SNES, SNES*);
extern PetscErrorCode SNESFASCycleGetInterpolation(SNES, Mat*);
extern PetscErrorCode SNESFASCycleGetRestriction(SNES, Mat*);
extern PetscErrorCode SNESFASCycleGetInjection(SNES, Mat*);
extern PetscErrorCode SNESFASCycleGetRScale(SNES, Vec*);
extern PetscErrorCode SNESFASCycleSetCycles(SNES, PetscInt);
extern PetscErrorCode SNESFASCycleIsFine(SNES, PetscBool*);

/* called on the (outer) finest level FAS to set/get parameters on any level instance */
extern PetscErrorCode SNESFASSetInterpolation(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASGetInterpolation(SNES, PetscInt, Mat*);
extern PetscErrorCode SNESFASSetRestriction(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASGetRestriction(SNES, PetscInt, Mat*);
extern PetscErrorCode SNESFASSetInjection(SNES, PetscInt, Mat);
extern PetscErrorCode SNESFASGetInjection(SNES, PetscInt, Mat*);
extern PetscErrorCode SNESFASSetRScale(SNES, PetscInt, Vec);
extern PetscErrorCode SNESFASGetRScale(SNES, PetscInt, Vec*);

extern PetscErrorCode SNESFASGetSmoother(SNES,     PetscInt, SNES*);
extern PetscErrorCode SNESFASGetSmootherUp(SNES,   PetscInt, SNES*);
extern PetscErrorCode SNESFASGetSmootherDown(SNES, PetscInt, SNES*);
extern PetscErrorCode SNESFASGetCoarseSolve(SNES, SNES*);

extern PetscErrorCode SNESFASCreateCoarseVec(SNES,Vec*);
extern PetscErrorCode SNESFASRestrict(SNES,Vec,Vec);

#endif
