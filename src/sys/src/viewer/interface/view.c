#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: view.c,v 1.33 1999/10/13 20:36:28 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy"
/*@C
   ViewerDestroy - Destroys a viewer.

   Collective on Viewer

   Input Parameters:
.  viewer - the viewer to be destroyed.

   Level: beginner

.seealso: ViewerSocketOpen(), ViewerASCIIOpen(), ViewerCreate(), ViewerDrawOpen()

.keywords: Viewer, destroy
@*/
int ViewerDestroy(Viewer viewer)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  if (--viewer->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(viewer);CHKERRQ(ierr);

  if (viewer->ops->destroy) {
    ierr = (*viewer->ops->destroy)(viewer);CHKERRQ(ierr);
  }
  PLogObjectDestroy((PetscObject)viewer);
  PetscHeaderDestroy((PetscObject)viewer);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerGetType"
/*@C
   ViewerGetType - Returns the type of a viewer.

   Not Collective

   Input Parameter:
.   viewer - the viewer

   Output Parameter:
.  type - viewer type (see below)

   Available Types Include:
.  SOCKET_VIEWER - Socket viewer
.  ASCII_VIEWER - ASCII viewer
.  BINARY_VIEWER - binary file viewer
.  STRING_VIEWER - string viewer
.  DRAW_VIEWER - drawing viewer

   Level: intermediate

   Note:
   See include/viewer.h for a complete list of viewers.

   ViewerType is actually a string

.keywords: Viewer, get, type

.seealso: ViewerCreate(), ViewerSetType()

@*/
int ViewerGetType(Viewer viewer,ViewerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  *type = (ViewerType) viewer->type_name;
  PetscFunctionReturn(0);
}






