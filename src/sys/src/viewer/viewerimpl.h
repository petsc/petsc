/* $Id: viewerimpl.h,v 1.4 2001/01/15 21:43:05 bsmith Exp balay $ */

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
  PetscViewerFormatType format,formats[10];
  int                   iformat;
  void                  *data;
};



#endif
