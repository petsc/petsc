#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: view.c,v 1.22 1998/03/06 00:18:33 bsmith Exp bsmith $";
#endif

#include "petsc.h" /*I "petsc.h" I*/

struct _p_Viewer {
   PETSCHEADER(int)
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
  int         ierr;
  PetscObject o = (PetscObject) v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (--v->refct > 0) PetscFunctionReturn(0);
  ierr = (*o->destroy)(o);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  *type = (ViewerType) v->type;
  PetscFunctionReturn(0);
}
