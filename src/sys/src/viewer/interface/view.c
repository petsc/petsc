#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: view.c,v 1.32 1999/09/02 14:52:46 bsmith Exp bsmith $";
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
int ViewerDestroy(Viewer v)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (--v->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

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
int ViewerGetType(Viewer v,ViewerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  *type = (ViewerType) v->type_name;
  PetscFunctionReturn(0);
}






