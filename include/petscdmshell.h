#if !defined(__PETSCDMSHELL_H)
#define __PETSCDMSHELL_H

#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode DMShellCreate(MPI_Comm,DM*);
extern PetscErrorCode DMShellSetMatrix(DM,Mat);
extern PetscErrorCode DMShellSetGlobalVector(DM,Vec);
extern PetscErrorCode DMShellSetCreateGlobalVector(DM,PetscErrorCode (*)(DM,Vec*));
extern PetscErrorCode DMShellSetCreateMatrix(DM,PetscErrorCode (*)(DM,const MatType,Mat*));

PETSC_EXTERN_CXX_END
#endif
