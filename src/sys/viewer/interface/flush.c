#define PETSC_DLL

#include "private/viewerimpl.h"  /*I "petscviewer.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFlush" 
/*@
   PetscViewerFlush - Flushes a PetscViewer (i.e. tries to dump all the 
   data that has been printed through a PetscViewer).

   Collective on PetscViewer

   Input Parameter:
.  viewer - the PetscViewer to be flushed

   Level: intermediate

   Concepts: flushing^Viewer data
   Concepts: redrawing^flushing 

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerCreate(), PetscViewerDestroy(),
          PetscViewerSetType()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerFlush(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  if (viewer->ops->flush) {
    ierr = (*viewer->ops->flush)(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


