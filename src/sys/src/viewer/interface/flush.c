/*$Id: flush.c,v 1.22 1999/10/24 14:01:08 bsmith Exp bsmith $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "viewer.h" I*/

#undef __FUNC__  
#define __FUNC__ "ViewerFlush"
/*@
   ViewerFlush - Flushes a viewer (i.e. tries to dump all the 
   data that has been printed through a viewer).

   Collective on Viewer

   Input Parameter:
.  viewer - the viewer to be flushed

   Level: intermediate

.keywords: Viewer, flush

.seealso: ViewerSocketOpen(), ViewerASCIIOpen(), ViewerDrawOpen(), ViewerCreate(), ViewerDestroy(),
          ViewerSetType()
@*/
int ViewerFlush(Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  if (viewer->ops->flush) {
    ierr = (*viewer->ops->flush)(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


