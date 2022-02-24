/* Routines to visualize DMs through GLVis */

#include <petsc/private/dmimpl.h>
#include <petsc/private/glvisviewerimpl.h>

PetscErrorCode DMView_GLVis(DM dm, PetscViewer viewer, PetscErrorCode (*DMView_GLVis_ASCII)(DM,PetscViewer))
{
  PetscBool      isglvis,isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  PetscCheck(isglvis || isascii,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer must be of type VIEWERGLVIS or VIEWERASCII");
  if (isglvis) {
    PetscViewerGLVisType type;
    PetscViewer          view;

    CHKERRQ(PetscViewerGLVisGetType_Private(viewer,&type));
    CHKERRQ(PetscViewerGLVisGetDMWindow_Private(viewer,&view));
    if (!view) PetscFunctionReturn(0); /* socket window has been closed */
    if (type == PETSC_VIEWER_GLVIS_SOCKET) {
      PetscMPIInt size,rank;
      PetscInt    sdim;
      const char* name;

      CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size));
      CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
      CHKERRQ(DMGetCoordinateDim(dm,&sdim));
      CHKERRQ(PetscObjectGetName((PetscObject)dm,&name));

      CHKERRQ(PetscGLVisCollectiveBegin(PetscObjectComm((PetscObject)dm),&view));
      CHKERRQ(PetscViewerASCIIPrintf(view,"parallel %d %d\nmesh\n",size,rank));
      CHKERRQ(DMView_GLVis_ASCII(dm,view));
      CHKERRQ(PetscViewerGLVisInitWindow_Private(view,PETSC_TRUE,sdim,name));
      CHKERRQ(PetscGLVisCollectiveEnd(PetscObjectComm((PetscObject)dm),&view));
    } else {
      CHKERRQ(DMView_GLVis_ASCII(dm,view));
    }
    CHKERRQ(PetscViewerGLVisRestoreDMWindow_Private(viewer,&view));
  } else {
    CHKERRQ(DMView_GLVis_ASCII(dm,viewer));
  }
  PetscFunctionReturn(0);
}
