/* $Id: viewerimpl.h,v 1.1 1999/05/12 03:33:52 bsmith Exp bsmith $ */

#ifndef _VIEWERIMPL
#define _VIEWERIMPL

#include "petsc.h"

struct _ViewerOps {
   int   (*destroy)(Viewer);
   int   (*view)(Viewer,Viewer);
   int   (*flush)(Viewer); 
   int   (*getsingleton)(Viewer,Viewer*);
   int   (*restoresingleton)(Viewer,Viewer*);
};

/*
   Defines the viewer data structure.
*/
struct _p_Viewer {
  PETSCHEADER(struct _ViewerOps)
  int   format,formats[10],iformat;
  char  *outputname,*outputnames[10];
  void *data;
};



#endif
