/* "Unintrusive" multi-physics DM */
#if !defined(__PETSCDMCOMPOSITE_H)
#define __PETSCDMCOMPOSITE_H

#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode   DMCompositeCreate(MPI_Comm,DM*);
extern PetscErrorCode   DMCompositeAddDM(DM,DM);
extern PetscErrorCode   DMCompositeSetCoupling(DM,PetscErrorCode (*)(DM,Mat,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,PetscInt));
extern PetscErrorCode   DMCompositeAddVecScatter(DM,VecScatter);
extern PetscErrorCode   DMCompositeScatter(DM,Vec,...);
extern PetscErrorCode   DMCompositeGather(DM,Vec,InsertMode,...);
extern PetscErrorCode   DMCompositeGetAccess(DM,Vec,...);
extern PetscErrorCode   DMCompositeGetNumberDM(DM,PetscInt*);
extern PetscErrorCode   DMCompositeRestoreAccess(DM,Vec,...);
extern PetscErrorCode   DMCompositeGetLocalVectors(DM,...);
extern PetscErrorCode   DMCompositeGetEntries(DM,...);
extern PetscErrorCode   DMCompositeGetEntriesArray(DM,DM[]);
extern PetscErrorCode   DMCompositeRestoreLocalVectors(DM,...);
extern PetscErrorCode   DMCompositeGetGlobalISs(DM,IS*[]);
extern PetscErrorCode   DMCompositeGetLocalISs(DM,IS**);
extern PetscErrorCode   DMCompositeGetISLocalToGlobalMappings(DM,ISLocalToGlobalMapping**);

PETSC_EXTERN_CXX_END
#endif
