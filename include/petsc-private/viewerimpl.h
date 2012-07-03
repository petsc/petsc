
#ifndef _VIEWERIMPL
#define _VIEWERIMPL

#include <petscsys.h>

struct _PetscViewerOps {
   PetscErrorCode (*destroy)(PetscViewer);
   PetscErrorCode (*view)(PetscViewer,PetscViewer);
   PetscErrorCode (*flush)(PetscViewer); 
   PetscErrorCode (*getsingleton)(PetscViewer,PetscViewer*);
   PetscErrorCode (*restoresingleton)(PetscViewer,PetscViewer*);
   PetscErrorCode (*getsubcomm)(PetscViewer,MPI_Comm,PetscViewer*);
   PetscErrorCode (*restoresubcomm)(PetscViewer,MPI_Comm,PetscViewer*);
   PetscErrorCode (*setfromoptions)(PetscViewer);
};

/*
   Defines the viewer data structure.
*/
struct _p_PetscViewer {
  PETSCHEADER(struct _PetscViewerOps);
  PetscViewerFormat format,formats[10];
  int               iformat;
  void              *data;
};



#endif
