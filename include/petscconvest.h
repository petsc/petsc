/*
      Objects which encapsulate discretizations+continuum residuals
*/
#if !defined(PETSCCE_H)
#define PETSCCE_H
#include <petscsnes.h>

/*S
  PetscConvEst - Provides an estimated convergence rate for a discretized problem

  Level: developer

.seealso: `PetscConvEstCreate()`, `PetscConvEstDestroy()`
S*/
typedef struct _p_PetscConvEst *PetscConvEst;

PETSC_EXTERN PetscErrorCode PetscConvEstCreate(MPI_Comm, PetscConvEst *);
PETSC_EXTERN PetscErrorCode PetscConvEstDestroy(PetscConvEst *);
PETSC_EXTERN PetscErrorCode PetscConvEstView(PetscConvEst, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscConvEstSetFromOptions(PetscConvEst);
PETSC_EXTERN PetscErrorCode PetscConvEstGetSolver(PetscConvEst, PetscObject *);
PETSC_EXTERN PetscErrorCode PetscConvEstSetSolver(PetscConvEst, PetscObject);
PETSC_EXTERN PetscErrorCode PetscConvEstSetUp(PetscConvEst);
PETSC_EXTERN PetscErrorCode PetscConvEstComputeInitialGuess(PetscConvEst, PetscInt, DM, Vec);
PETSC_EXTERN PetscErrorCode PetscConvEstComputeError(PetscConvEst, PetscInt, DM, Vec, PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscConvEstGetConvRate(PetscConvEst, PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscConvEstMonitorDefault(PetscConvEst, PetscInt);
PETSC_EXTERN PetscErrorCode PetscConvEstRateView(PetscConvEst, const PetscReal[], PetscViewer);

#endif
