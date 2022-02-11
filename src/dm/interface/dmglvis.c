/* Routines to visualize DMs through GLVis */

#include <petsc/private/dmimpl.h>
#include <petsc/private/glvisviewerimpl.h>

PetscErrorCode DMView_GLVis(DM dm, PetscViewer viewer, PetscErrorCode (*DMView_GLVis_ASCII)(DM,PetscViewer))
{
  PetscErrorCode ierr;
  PetscBool      isglvis,isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&isglvis);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  PetscCheckFalse(!isglvis && !isascii,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer must be of type VIEWERGLVIS or VIEWERASCII");
  if (isglvis) {
    PetscViewerGLVisType type;
    PetscViewer          view;

    ierr = PetscViewerGLVisGetType_Private(viewer,&type);CHKERRQ(ierr);
    ierr = PetscViewerGLVisGetDMWindow_Private(viewer,&view);CHKERRQ(ierr);
    if (!view) PetscFunctionReturn(0); /* socket window has been closed */
    if (type == PETSC_VIEWER_GLVIS_SOCKET) {
      PetscMPIInt size,rank;
      PetscInt    sdim;
      const char* name;

      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRMPI(ierr);
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
      ierr = DMGetCoordinateDim(dm,&sdim);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject)dm,&name);CHKERRQ(ierr);

      ierr = PetscGLVisCollectiveBegin(PetscObjectComm((PetscObject)dm),&view);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(view,"parallel %d %d\nmesh\n",size,rank);CHKERRQ(ierr);
      ierr = DMView_GLVis_ASCII(dm,view);CHKERRQ(ierr);
      ierr = PetscViewerGLVisInitWindow_Private(view,PETSC_TRUE,sdim,name);CHKERRQ(ierr);
      ierr = PetscGLVisCollectiveEnd(PetscObjectComm((PetscObject)dm),&view);CHKERRQ(ierr);
    } else {
      ierr = DMView_GLVis_ASCII(dm,view);CHKERRQ(ierr);
    }
    ierr = PetscViewerGLVisRestoreDMWindow_Private(viewer,&view);CHKERRQ(ierr);
  } else {
    ierr = DMView_GLVis_ASCII(dm,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
