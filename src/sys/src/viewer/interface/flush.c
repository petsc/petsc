#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: flush.c,v 1.15 1998/04/13 17:55:21 bsmith Exp bsmith $";
#endif

#include "pinclude/pviewer.h"  /*I "viewer.h" I*/

struct _p_Viewer {
  VIEWERHEADER
};

#undef __FUNC__  
#define __FUNC__ "ViewerFlush"
/*@
   ViewerFlush - Flushes a viewer (i.e. tries to dump all the 
   data that has been printed through a viewer).

   Input Parameters:
.  viewer - the viewer to be flushed

   Collective on Viewer

.keywords: Viewer, flush

.seealso: ViewerMatlabOpen(), ViewerFileOpenASCII()
@*/
int ViewerFlush(Viewer v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->flush) {
    ierr = (*v->flush)(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


