#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: view.c,v 1.25 1998/04/13 17:55:15 bsmith Exp curfman $";
#endif

#include "petsc.h" /*I "petsc.h" I*/
#include "pinclude/pviewer.h"

struct _p_Viewer {
   VIEWERHEADER
};

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy"
/*@C
   ViewerDestroy - Destroys a viewer.

   Collective on Viewer

   Input Parameters:
.  viewer - the viewer to be destroyed.

.seealso: ViewerMatlabOpen(), ViewerFileOpenASCII()

.keywords: Viewer, destroy
@*/
int ViewerDestroy(Viewer v)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (--v->refct > 0) PetscFunctionReturn(0);
  ierr = (*v->destroy)(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerGetType"
/*@
   ViewerGetType - Returns the type of a viewer.

   Not Collective

   Input Parameter:
   v - the viewer

   Output Parameter:
.  type - viewer type (see below)

   Available Types Include:
.  MATLAB_VIEWER - Matlab viewer
.  ASCII_FILE_VIEWER - uniprocess ASCII viewer
.  ASCII_FILES_VIEWER - parallel ASCII viewer
.  BINARY_FILE_VIEWER - binary file viewer
.  STRING_VIEWER - string viewer
.  DRAW_VIEWER - drawing viewer

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
