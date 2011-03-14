/* Very minimal unstructured DM */
#if !defined(__PETSCDMSLICED_H)
#define __PETSCDMSLICED_H

#include "petscdm.h"
extern PetscErrorCode   DMSlicedCreate(MPI_Comm,DM*);
extern PetscErrorCode   DMSlicedGetGlobalIndices(DM,PetscInt*[]);
extern PetscErrorCode   DMSlicedSetPreallocation(DM,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
extern PetscErrorCode   DMSlicedSetBlockFills(DM,const PetscInt*,const PetscInt*);
extern PetscErrorCode   DMSlicedSetGhosts(DM,PetscInt,PetscInt,PetscInt,const PetscInt[]);

#endif
