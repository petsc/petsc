#ifndef __TSHISTORYIMPL_H
#define __TSHISTORYIMPL_H

#include <petsc/private/tsimpl.h>

PETSC_INTERN PetscErrorCode TSHistoryCreate(MPI_Comm,TSHistory*);
PETSC_INTERN PetscErrorCode TSHistoryDestroy(TSHistory*);
PETSC_INTERN PetscErrorCode TSHistorySetHistory(TSHistory,PetscInt,PetscReal[],PetscInt[],PetscBool);
PETSC_INTERN PetscErrorCode TSHistoryGetHistory(TSHistory,PetscInt*,const PetscReal*[],const PetscInt*[],PetscBool*);
PETSC_INTERN PetscErrorCode TSHistoryGetLocFromTime(TSHistory,PetscReal,PetscInt*);
PETSC_INTERN PetscErrorCode TSHistoryUpdate(TSHistory,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TSHistoryGetTimeStep(TSHistory,PetscBool,PetscInt,PetscReal*);
PETSC_INTERN PetscErrorCode TSHistoryGetNumSteps(TSHistory,PetscInt*);
#endif
