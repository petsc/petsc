
#if !defined(__PETSCVIEWERAMS_H)
#define __PETSCVIEWERAMS_H

#include <petscviewer.h>
#include <ams.h>
PETSC_EXTERN PetscErrorCode PetscViewerAMSSetCommName(PetscViewer,const char[]);
PETSC_EXTERN PetscErrorCode PetscViewerAMSGetAMSComm(PetscViewer,AMS_Comm *);
PETSC_EXTERN PetscErrorCode PetscViewerAMSOpen(MPI_Comm,const char[],PetscViewer*);
PETSC_EXTERN PetscViewer    PETSC_VIEWER_AMS_(MPI_Comm);
PETSC_EXTERN PetscErrorCode PETSC_VIEWER_AMS_Destroy(MPI_Comm);
#define PETSC_VIEWER_AMS_WORLD PETSC_VIEWER_AMS_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_AMS_SELF  PETSC_VIEWER_AMS_(PETSC_COMM_SELF)

#endif
