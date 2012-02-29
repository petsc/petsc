/* Very minimal unstructured DM */
#if !defined(__PETSCDMSLICED_H)
#define __PETSCDMSLICED_H

#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode   DMSlicedCreate(MPI_Comm,DM*);
extern PetscErrorCode   DMSlicedSetPreallocation(DM,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
extern PetscErrorCode   DMSlicedSetBlockFills(DM,const PetscInt*,const PetscInt*);
extern PetscErrorCode   DMSlicedSetGhosts(DM,PetscInt,PetscInt,PetscInt,const PetscInt[]);

PETSC_EXTERN_CXX_END
#endif
