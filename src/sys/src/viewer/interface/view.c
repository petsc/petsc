#ifndef lint
static char vcid[] = "$Id: view.c,v 1.8 1995/10/01 21:53:22 bsmith Exp bsmith $";
#endif

#include "petsc.h"

struct _Viewer {
   PETSCHEADER
};

/*@C
   ViewerDestroy - Destroys a viewer.

   Input Parameters:
.  viewer - the viewer to be destroyed.

.seealso: ViewerMatlabOpen(), ViewerFileOpenASCII()

.keywords: Viewer, destroy
@*/
int ViewerDestroy(Viewer v)
{
  PetscObject o = (PetscObject) v;
  PETSCVALIDHEADERSPECIFIC(v,VIEWER_COOKIE);
  return (*o->destroy)(o);
}

/*@
    ViewerGetType - Returns the type of a viewer.

  Input Parameter:
   v - the viewer

  Output Parameter:
   type - one of MATLAB_VIEWER, ASCII_FILE_VIEWER, ASCII_FILES_VIEWER,
                 BINARY_FILE_VIEWER, STRING_VIEWER, ...

@*/
int ViewerGetType(Viewer v,ViewerType *type)
{
  PETSCVALIDHEADERSPECIFIC(v,VIEWER_COOKIE);
  *type = (ViewerType) v->type;
  return 0;
}
