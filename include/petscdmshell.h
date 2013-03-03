#if !defined(__PETSCDMSHELL_H)
#define __PETSCDMSHELL_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMShellCreate(MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMShellSetMatrix(DM,Mat);
PETSC_EXTERN PetscErrorCode DMShellSetGlobalVector(DM,Vec);
PETSC_EXTERN PetscErrorCode DMShellSetLocalVector(DM,Vec);
PETSC_EXTERN PetscErrorCode DMShellSetCreateGlobalVector(DM,PetscErrorCode (*)(DM,Vec*));
PETSC_EXTERN PetscErrorCode DMShellSetCreateLocalVector(DM,PetscErrorCode (*)(DM,Vec*));
PETSC_EXTERN PetscErrorCode DMShellSetGlobalToLocal(DM,PetscErrorCode (*)(DM,Vec,InsertMode,Vec),PetscErrorCode (*)(DM,Vec,InsertMode,Vec));
PETSC_EXTERN PetscErrorCode DMShellSetLocalToGlobal(DM,PetscErrorCode (*)(DM,Vec,InsertMode,Vec),PetscErrorCode (*)(DM,Vec,InsertMode,Vec));
PETSC_EXTERN PetscErrorCode DMShellSetCreateMatrix(DM,PetscErrorCode (*)(DM,MatType,Mat*));
PETSC_EXTERN PetscErrorCode DMShellDefaultGlobalToLocalBegin(DM dm,Vec g,InsertMode mode,Vec l);
PETSC_EXTERN PetscErrorCode DMShellDefaultGlobalToLocalEnd(DM dm,Vec g,InsertMode mode,Vec l);

#endif
