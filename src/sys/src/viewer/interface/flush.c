/*$Id: flush.c,v 1.28 2000/09/22 20:41:53 bsmith Exp bsmith $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petscviewer.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=ViewerFlush""></a>*/"PetscViewerFlush" 
/*@
   PetscViewerFlush - Flushes a PetscViewer (i.e. tries to dump all the 
   data that has been printed through a PetscViewer).

   Collective on PetscViewer

   Input Parameter:
.  PetscViewer - the PetscViewer to be flushed

   Level: intermediate

   Concepts: flushing^Viewer data
   Concepts: redrawing^flushing 

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerCreate(), PetscViewerDestroy(),
          PetscViewerSetType()
@*/
int PetscViewerFlush(PetscViewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  if (viewer->ops->flush) {
    ierr = (*viewer->ops->flush)(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


