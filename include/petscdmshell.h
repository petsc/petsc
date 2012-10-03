#if !defined(__PETSCDMSHELL_H)
#define __PETSCDMSHELL_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMShellCreate(MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMShellSetMatrix(DM,Mat);
PETSC_EXTERN PetscErrorCode DMShellSetGlobalVector(DM,Vec);
PETSC_EXTERN PetscErrorCode DMShellSetCreateGlobalVector(DM,PetscErrorCode (*)(DM,Vec*));
PETSC_EXTERN PetscErrorCode DMShellSetCreateMatrix(DM,PetscErrorCode (*)(DM,MatType,Mat*));

#endif
