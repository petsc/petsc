
#include <petsc/private/viewerimpl.h>  /*I "petscviewer.h" I*/

/*@
   PetscViewerFlush - Flushes a PetscViewer (i.e. tries to dump all the
   data that has been printed through a PetscViewer).

   Collective on PetscViewer

   Input Parameter:
.  viewer - the PetscViewer to be flushed

   Level: intermediate

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerCreate(), PetscViewerDestroy(),
          PetscViewerSetType()
@*/
PetscErrorCode  PetscViewerFlush(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (viewer->ops->flush) {
    ierr = (*viewer->ops->flush)(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

