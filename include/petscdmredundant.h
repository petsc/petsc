/* DM for redundant globally coupled degrees of freedom */
#if !defined(__PETSCDMREDUNDANT_H)
#define __PETSCDMREDUNDANT_H

#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode DMRedundantCreate(MPI_Comm,PetscInt,PetscInt,DM*);
extern PetscErrorCode DMRedundantSetSize(DM,PetscInt,PetscInt);
extern PetscErrorCode DMRedundantGetSize(DM,PetscInt*,PetscInt*);

PETSC_EXTERN_CXX_END
#endif
