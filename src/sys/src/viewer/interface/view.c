#ifndef lint
static char vcid[] = "$Id: view.c,v 1.15 1996/12/16 21:05:12 balay Exp balay $";
#endif

#include "petsc.h" /*I "petsc.h" I*/

struct _Viewer {
   PETSCHEADER
   int         (*flush)(Viewer);
};

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy"
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
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  return (*o->destroy)(o);
}

#undef __FUNC__  
#define __FUNC__ "ViewerGetType"
/*@
   ViewerGetType - Returns the type of a viewer.

   Input Parameter:
   v - the viewer

   Output Parameter:
.  type - one of
$    MATLAB_VIEWER,
$    ASCII_FILE_VIEWER,
$    ASCII_FILES_VIEWER,
$    BINARY_FILE_VIEWER,
$    STRING_VIEWER,
$    DRAW_VIEWER, ...

   Note:
   See petsc/include/viewer.h for a complete list of viewers.

.keywords: Viewer, get, type
@*/
int ViewerGetType(Viewer v,ViewerType *type)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  *type = (ViewerType) v->type;
  return 0;
}
