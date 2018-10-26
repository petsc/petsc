
#ifndef _VIEWERIMPL
#define _VIEWERIMPL

#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

PETSC_EXTERN PetscBool      PetscViewerRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscViewerRegisterAll(void);

struct _PetscViewerOps {
   PetscErrorCode (*destroy)(PetscViewer);
   PetscErrorCode (*view)(PetscViewer,PetscViewer);
   PetscErrorCode (*flush)(PetscViewer);
   PetscErrorCode (*getsubviewer)(PetscViewer,MPI_Comm,PetscViewer*);
   PetscErrorCode (*restoresubviewer)(PetscViewer,MPI_Comm,PetscViewer*);
   PetscErrorCode (*read)(PetscViewer,void*,PetscInt,PetscInt*,PetscDataType);
   PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscViewer);
   PetscErrorCode (*setup)(PetscViewer);
};

#define PETSCVIEWERGETVIEWEROFFPUSHESMAX 25

#define PETSCVIEWERFORMATPUSHESMAX 25
/*
   Defines the viewer data structure.
*/
struct _p_PetscViewer {
  PETSCHEADER(struct _PetscViewerOps);
  PetscViewerFormat format,formats[PETSCVIEWERFORMATPUSHESMAX];
  int               iformat;   /* number of formats that have been pushed on formats[] stack */
  void              *data;
  PetscBool         setupcalled;
};

PETSC_EXTERN PetscMPIInt Petsc_Viewer_keyval;
PETSC_EXTERN PetscMPIInt Petsc_Viewer_Stdout_keyval;
PETSC_EXTERN PetscMPIInt Petsc_Viewer_Stderr_keyval;
PETSC_EXTERN PetscMPIInt Petsc_Viewer_Binary_keyval;
PETSC_EXTERN PetscMPIInt Petsc_Viewer_Draw_keyval;
#if defined(PETSC_HAVE_HDF5)
PETSC_EXTERN PetscMPIInt Petsc_Viewer_HDF5_keyval;
#endif
#if defined(PETSC_USE_SOCKETVIEWER)
PETSC_EXTERN PetscMPIInt Petsc_Viewer_Socket_keyval;
#endif

#if defined(PETSC_HAVE_HDF5)
#include <petscviewerhdf5.h>
struct _n_HDF5ReadCtx {
  hid_t file, group, dataset, dataspace;
};
typedef struct _n_HDF5ReadCtx* HDF5ReadCtx;
PETSC_INTERN PetscErrorCode PetscViewerHDF5ReadInitialize_Internal(PetscViewer,const char[],HDF5ReadCtx*,PetscInt*,PetscBool*);
PETSC_INTERN PetscErrorCode PetscViewerHDF5ReadFinalize_Internal(PetscViewer,HDF5ReadCtx*);
PETSC_INTERN PetscErrorCode PetscViewerHDF5ReadSizes_Internal(HDF5ReadCtx,PetscInt,PetscBool,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*);
#endif

#endif
