/*$Id: flush.c,v 1.24 2000/04/09 04:34:00 bsmith Exp bsmith $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "viewer.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ViewerFlush" 
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


