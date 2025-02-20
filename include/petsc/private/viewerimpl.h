#pragma once

#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

PETSC_INTERN PetscBool      PetscViewerRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscViewerRegisterAll(void);

struct _PetscViewerOps {
  PetscErrorCode (*destroy)(PetscViewer);
  PetscErrorCode (*view)(PetscViewer, PetscViewer);
  PetscErrorCode (*flush)(PetscViewer);
  PetscErrorCode (*getsubviewer)(PetscViewer, MPI_Comm, PetscViewer *);
  PetscErrorCode (*restoresubviewer)(PetscViewer, MPI_Comm, PetscViewer *);
  PetscErrorCode (*read)(PetscViewer, void *, PetscInt, PetscInt *, PetscDataType);
  PetscErrorCode (*setfromoptions)(PetscViewer, PetscOptionItems);
  PetscErrorCode (*setup)(PetscViewer);
};

#define PETSCVIEWERCREATEVIEWEROFFPUSHESMAX 25

#define PETSCVIEWERFORMATPUSHESMAX 25
/*
   Defines the viewer data structure.
*/
typedef struct _PetscViewerOps *PetscViewerOps;
struct _p_PetscViewer {
  PETSCHEADER(struct _PetscViewerOps);
  PetscViewerFormat format, formats[PETSCVIEWERFORMATPUSHESMAX];
  int               iformat; /* number of formats that have been pushed on formats[] stack */
  void             *data;
  PetscBool         setupcalled;
};

PETSC_INTERN PetscMPIInt Petsc_Viewer_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Stdout_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Stderr_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Binary_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Draw_keyval;
#if defined(PETSC_HAVE_HDF5)
PETSC_INTERN PetscMPIInt Petsc_Viewer_HDF5_keyval;
#endif
#if defined(PETSC_USE_SOCKETVIEWER)
PETSC_INTERN PetscMPIInt Petsc_Viewer_Socket_keyval;
#endif
