/*$Id: flush.c,v 1.29 2001/01/15 21:43:19 bsmith Exp balay $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petscviewer.h" I*/

#undef __FUNCT__  
#define __FUNCT__ /*<a name=ViewerFlush""></a>*/"PetscViewerFlush" 
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


