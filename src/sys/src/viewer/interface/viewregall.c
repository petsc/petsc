#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: view.c,v 1.27 1998/12/03 04:05:14 bsmith Exp $";
#endif

#include "src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy"
/*@C
   ViewerDestroy - Destroys a viewer.

   Collective on Viewer

   Input Parameters:
.  viewer - the viewer to be destroyed.

.seealso: ViewerMatlabOpen(), ViewerASCIIOpen()

.keywords: Viewer, destroy
@*/
int ViewerDestroy(Viewer v)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (--v->refct > 0) PetscFunctionReturn(0);
  if (v->ops->destroy) {
    ierr = (*v->ops->destroy)(v);CHKERRQ(ierr);
  }
  PLogObjectDestroy((PetscObject)v);
  PetscHeaderDestroy((PetscObject)v);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerGetType"
/*@C
   ViewerGetType - Returns the type of a viewer.

   Not Collective

   Input Parameter:
.   v - the viewer

   Output Parameter:
.  type - viewer type (see below)

   Available Types Include:
.  MATLAB_VIEWER - Matlab viewer
.  ASCII_VIEWER - ASCII viewer
.  BINARY_VIEWER - binary file viewer
.  STRING_VIEWER - string viewer
.  DRAW_VIEWER - drawing viewer

   Note:
   See petsc/include/viewer.h for a complete list of viewers.

   ViewerType is actually a string

.keywords: Viewer, get, type
@*/
int ViewerGetType(Viewer v,ViewerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  *type = (ViewerType) v->type_name;
  PetscFunctionReturn(0);
}






