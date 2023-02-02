/* Routines to visualize DMs through GLVis */

#include <petsc/private/dmimpl.h>
#include <petsc/private/glvisviewerimpl.h>

PetscErrorCode DMView_GLVis(DM dm, PetscViewer viewer, PetscErrorCode (*DMView_GLVis_ASCII)(DM, PetscViewer))
{
  PetscBool isglvis, isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERGLVIS, &isglvis));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isglvis || isascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer must be of type VIEWERGLVIS or VIEWERASCII");
  if (isglvis) {
    PetscViewerGLVisType type;
    PetscViewer          view;

    PetscCall(PetscViewerGLVisGetType_Private(viewer, &type));
    PetscCall(PetscViewerGLVisGetDMWindow_Private(viewer, &view));
    if (!view) PetscFunctionReturn(PETSC_SUCCESS); /* socket window has been closed */
    if (type == PETSC_VIEWER_GLVIS_SOCKET) {
      PetscMPIInt size, rank;
      PetscInt    sdim;
      const char *name;

      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
      PetscCall(DMGetCoordinateDim(dm, &sdim));
      PetscCall(PetscObjectGetName((PetscObject)dm, &name));

      PetscCall(PetscGLVisCollectiveBegin(PetscObjectComm((PetscObject)dm), &view));
      PetscCall(PetscViewerASCIIPrintf(view, "parallel %d %d\nmesh\n", size, rank));
      PetscCall(DMView_GLVis_ASCII(dm, view));
      PetscCall(PetscViewerGLVisInitWindow_Private(view, PETSC_TRUE, sdim, name));
      PetscCall(PetscGLVisCollectiveEnd(PetscObjectComm((PetscObject)dm), &view));
    } else {
      PetscCall(DMView_GLVis_ASCII(dm, view));
    }
    PetscCall(PetscViewerGLVisRestoreDMWindow_Private(viewer, &view));
  } else {
    PetscCall(DMView_GLVis_ASCII(dm, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
