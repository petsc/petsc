
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



#endif
