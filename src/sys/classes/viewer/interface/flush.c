
#include <petsc/private/viewerimpl.h> /*I "petscviewer.h" I*/

/*@
   PetscViewerFlush - Flushes a `PetscViewer` (i.e. tries to dump all the
   data that has been printed through a `PetscViewer`).

   Collective on viewer

   Input Parameter:
.  viewer - the `PetscViewer` to be flushed

   Level: intermediate

.seealso: `PetscViewer`, `PetscViewerSocketOpen()`, `PetscViewerASCIIOpen()`, `PetscViewerDrawOpen()`, `PetscViewerCreate()`, `PetscViewerDestroy()`,
          `PetscViewerSetType()`
@*/
PetscErrorCode PetscViewerFlush(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryTypeMethod(viewer, flush);
  PetscFunctionReturn(0);
}
