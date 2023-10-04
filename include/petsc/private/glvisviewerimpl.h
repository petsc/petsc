#pragma once

#include <petscviewer.h>
#include <petscsys.h>

struct _n_PetscViewerGLVisVecInfo {
  char *fec_type; /* the output of FiniteElementCollection::Name() */
};
typedef struct _n_PetscViewerGLVisVecInfo *PetscViewerGLVisVecInfo;

struct _n_PetscViewerGLVisInfo {
  PetscBool enabled; /* whether or not to visualize data from the process (it works, but it currently misses a public API) */
  PetscBool init;    /* whether or not the popup window has been initialized (must be done after having sent the data the first time) */
  PetscInt  size[2]; /* window sizes */
  PetscReal pause;   /* pause argument */
  char     *fmt;     /* format */
};
typedef struct _n_PetscViewerGLVisInfo *PetscViewerGLVisInfo;

typedef enum {
  PETSCVIEWERGLVIS_DISCONNECTED,
  PETSCVIEWERGLVIS_CONNECTED,
  PETSCVIEWERGLVIS_DISABLED
} PetscViewerGLVisStatus;

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisPause_Internal(PetscViewer);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisSetDM_Internal(PetscViewer, PetscObject);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisGetDM_Internal(PetscViewer, PetscObject *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisInitWindow_Internal(PetscViewer, PetscBool, PetscInt, const char *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisGetStatus_Internal(PetscViewer, PetscViewerGLVisStatus *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisGetType_Internal(PetscViewer, PetscViewerGLVisType *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisGetWindow_Internal(PetscViewer, PetscInt, PetscViewer *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisRestoreWindow_Internal(PetscViewer, PetscInt, PetscViewer *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisGetFields_Internal(PetscViewer, PetscInt *, const char **[], PetscInt *[], PetscErrorCode (**)(PetscObject, PetscInt, PetscObject[], void *), PetscObject *[], void **);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisGetDMWindow_Internal(PetscViewer, PetscViewer *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerGLVisRestoreDMWindow_Internal(PetscViewer, PetscViewer *);

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscGLVisCollectiveBegin(MPI_Comm, PetscViewer *);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscGLVisCollectiveEnd(MPI_Comm, PetscViewer *);
