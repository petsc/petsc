
#if !defined(__PETSCVIEWERSAWS_H)
#define __PETSCVIEWERSAWS_H

#include <petscviewer.h>
#include <SAWs.h>
extern SAWs_Directory PETSC_SAWs_ROOT_DIRECTORY;
extern SAWs_Directory PETSC_OBJECTS_SAWs_ROOT_DIRECTORY;
PETSC_EXTERN PetscErrorCode PetscViewerSAWSOpen(MPI_Comm,PetscViewer*);
PETSC_EXTERN PetscViewer    PETSC_VIEWER_SAWS_(MPI_Comm);
PETSC_EXTERN PetscErrorCode PETSC_VIEWER_SAWS_Destroy(MPI_Comm);
#define PETSC_VIEWER_SAWS_WORLD PETSC_VIEWER_SAWS_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_SAWS_SELF  PETSC_VIEWER_SAWS_(PETSC_COMM_SELF)

#define PetscStackCallSAWs(func,args) do {PetscErrorCode _ierr; char* err; \
    PetscStackPush(#func);_ierr = func args;PetscStackPop; if (_ierr) {SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in %s() %s",#func,err);} \
} while (0)

#endif
