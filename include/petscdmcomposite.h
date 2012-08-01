/* "Unintrusive" multi-physics DM */
#if !defined(__PETSCDMCOMPOSITE_H)
#define __PETSCDMCOMPOSITE_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMCompositeCreate(MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMCompositeAddDM(DM,DM);
PETSC_EXTERN PetscErrorCode DMCompositeSetCoupling(DM,PetscErrorCode (*)(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt));
PETSC_EXTERN PetscErrorCode DMCompositeAddVecScatter(DM,VecScatter);
PETSC_EXTERN PetscErrorCode DMCompositeScatter(DM,Vec,...);
PETSC_EXTERN PetscErrorCode DMCompositeGather(DM,Vec,InsertMode,...);
PETSC_EXTERN PetscErrorCode DMCompositeGetAccess(DM,Vec,...);
PETSC_EXTERN PetscErrorCode DMCompositeGetNumberDM(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMCompositeRestoreAccess(DM,Vec,...);
PETSC_EXTERN PetscErrorCode DMCompositeGetLocalVectors(DM,...);
PETSC_EXTERN PetscErrorCode DMCompositeGetEntries(DM,...);
PETSC_EXTERN PetscErrorCode DMCompositeGetEntriesArray(DM,DM[]);
PETSC_EXTERN PetscErrorCode DMCompositeRestoreLocalVectors(DM,...);
PETSC_EXTERN PetscErrorCode DMCompositeGetGlobalISs(DM,IS*[]);
PETSC_EXTERN PetscErrorCode DMCompositeGetLocalISs(DM,IS**);
PETSC_EXTERN PetscErrorCode DMCompositeGetISLocalToGlobalMappings(DM,ISLocalToGlobalMapping**);

#endif
