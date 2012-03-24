#include <petsc-private/snesimpl.h>

extern PetscErrorCode SNESVIProjectOntoBounds(SNES,Vec);
extern PetscErrorCode SNESVICheckLocalMin_Private(SNES,Mat,Vec,Vec,PetscReal,PetscBool*);
extern PetscErrorCode SNESReset_VI(SNES);
extern PetscErrorCode SNESDestroy_VI(SNES);
extern PetscErrorCode SNESView_VI(SNES,PetscViewer);
extern PetscErrorCode SNESSetFromOptions_VI(SNES);
extern PetscErrorCode SNESSetUp_VI(SNES);
extern PetscErrorCode SNESDefaultConverged_VI(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
extern PetscErrorCode SNESSolve_VISS(SNES);

/* composed functions */
EXTERN_C_BEGIN
extern PetscErrorCode SNESVISetComputeVariableBounds_VI(SNES,SNESVIComputeVariableBoundsFunction);
extern PetscErrorCode SNESVISetVariableBounds_VI(SNES,Vec,Vec);
EXTERN_C_END
