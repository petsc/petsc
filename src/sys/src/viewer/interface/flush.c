#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: flush.c,v 1.16 1998/04/20 19:31:22 bsmith Exp bsmith $";
#endif

#include "src/viewer/viewerimpl.h"  /*I "viewer.h" I*/

#undef __FUNC__  
#define __FUNC__ "ViewerFlush"
/*@
   ViewerFlush - Flushes a viewer (i.e. tries to dump all the 
   data that has been printed through a viewer).

   Collective on Viewer

   Input Parameters:
.  viewer - the viewer to be flushed

.keywords: Viewer, flush

.seealso: ViewerMatlabOpen(), ViewerASCIIOpen()
@*/
int ViewerFlush(Viewer v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->ops->flush) {
    ierr = (*v->ops->flush)(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


