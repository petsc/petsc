
#ifndef _VIEWERIMPL
#define _VIEWERIMPL

#include <petsc-private/petscimpl.h>
#include <petscviewer.h>

PETSC_EXTERN PetscBool PetscViewerRegisterAllCalled;

struct _PetscViewerOps {
   PetscErrorCode (*destroy)(PetscViewer);
   PetscErrorCode (*view)(PetscViewer,PetscViewer);
   PetscErrorCode (*flush)(PetscViewer);
   PetscErrorCode (*getsingleton)(PetscViewer,PetscViewer*);
   PetscErrorCode (*restoresingleton)(PetscViewer,PetscViewer*);
   PetscErrorCode (*getsubcomm)(PetscViewer,MPI_Comm,PetscViewer*);
   PetscErrorCode (*restoresubcomm)(PetscViewer,MPI_Comm,PetscViewer*);
   PetscErrorCode (*setfromoptions)(PetscViewer);
   PetscErrorCode (*read)(PetscViewer,void*,PetscInt,PetscDataType);
};

/*
   Defines the viewer data structure.
*/
struct _p_PetscViewer {
  PETSCHEADER(struct _PetscViewerOps);
  PetscViewerFormat format,formats[10];
  int               iformat;   /* number of formats that have been pushed on formats[] stack */
  void              *data;
};



#endif
