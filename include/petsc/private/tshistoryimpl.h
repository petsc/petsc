#ifndef __TSHISTORYIMPL_H
#define __TSHISTORYIMPL_H

#include <petsc/private/tsimpl.h>

PETSC_EXTERN PetscErrorCode TSHistoryCreate(MPI_Comm,TSHistory*);
PETSC_EXTERN PetscErrorCode TSHistoryDestroy(TSHistory*);
PETSC_EXTERN PetscErrorCode TSHistorySetHistory(TSHistory,PetscInt,PetscReal[],PetscInt[],PetscBool);
PETSC_EXTERN PetscErrorCode TSHistoryGetHistory(TSHistory,PetscInt*,const PetscReal*[],const PetscInt*[],PetscBool*);
PETSC_EXTERN PetscErrorCode TSHistoryGetLocFromTime(TSHistory,PetscReal,PetscInt*);
PETSC_EXTERN PetscErrorCode TSHistoryUpdate(TSHistory,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode TSHistoryGetTimeStep(TSHistory,PetscBool,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode TSHistoryGetTime(TSHistory,PetscBool,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode TSHistoryGetNumSteps(TSHistory,PetscInt*);
#endif
