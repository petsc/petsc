
#ifndef _VIEWERIMPL
#define _VIEWERIMPL

#include "petsc.h"

struct _PetscViewerOps {
   int   (*destroy)(PetscViewer);
   int   (*view)(PetscViewer,PetscViewer);
   int   (*flush)(PetscViewer); 
   int   (*getsingleton)(PetscViewer,PetscViewer*);
   int   (*restoresingleton)(PetscViewer,PetscViewer*);
   int   (*setfromoptions)(PetscViewer);
};

/*
   Defines the viewer data structure.
*/
struct _p_PetscViewer {
  PETSCHEADER(struct _PetscViewerOps)
  PetscViewerFormat format,formats[10];
  int               iformat;
  void              *data;
};



#endif
