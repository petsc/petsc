/* $Id: viewerimpl.h,v 1.3 2000/09/02 02:46:26 bsmith Exp bsmith $ */

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
  int   format,formats[10],iformat;
  char  *outputname,*outputnames[10];
  void  *data;
};



#endif
