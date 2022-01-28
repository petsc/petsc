
#include <petsc/private/viewerimpl.h>  /*I "petscviewer.h" I*/

/*@C
   PetscViewerGetSubViewer - Creates a new PetscViewer (same type as the old)
    that lives on a subcommunicator

    Collective on PetscViewer

   Input Parameter:
.  viewer - the PetscViewer to be reproduced

   Output Parameter:
.  outviewer - new PetscViewer

   Level: advanced

   Notes:
    The output of the subviewers is synchronized against the original viewer. For example, if a
    viewer on two MPI ranks is decomposed into two subviewers, the output from the first viewer is
    all printed before the output from the second viewer. You must call PetscViewerFlush() after
    the call to PetscViewerRestoreSubViewer().

    Call PetscViewerRestoreSubViewer() to destroy this PetscViewer, NOT PetscViewerDestroy()

     This is most commonly used to view a sequential object that is part of a
    parallel object. For example block Jacobi PC view could use this to obtain a
    PetscViewer that is used with the sequential KSP on one block of the preconditioner.

    Between the calls to PetscViewerGetSubViewer() and PetscViewerRestoreSubViewer() the original
    viewer should not be used

    PETSCVIEWERDRAW and PETSCVIEWERBINARY only support returning a singleton viewer on rank 0,
    all other ranks will return a NULL viewer

  Developer Notes:
    There is currently incomplete error checking that the user does not use the original viewer between the
    the calls to PetscViewerGetSubViewer() and PetscViewerRestoreSubViewer(). If the user does there
    could be errors in the viewing that go undetected or crash the code.

    It would be nice if the call to PetscViewerFlush() was not required and was handled by
    PetscViewerRestoreSubViewer()

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerRestoreSubViewer()
@*/
PetscErrorCode  PetscViewerGetSubViewer(PetscViewer viewer,MPI_Comm comm,PetscViewer *outviewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(outviewer,3);
  if (viewer->ops->getsubviewer) {
    ierr = (*viewer->ops->getsubviewer)(viewer,comm,outviewer);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get SubViewer PetscViewer for type %s",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerRestoreSubViewer - Restores a new PetscViewer obtained with PetscViewerGetSubViewer().

    Collective on PetscViewer

   Input Parameters:
+  viewer - the PetscViewer that was reproduced
-  outviewer - new PetscViewer

   Level: advanced

   Notes:
    Call PetscViewerGetSubViewer() to get this PetscViewer, NOT PetscViewerCreate()

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerGetSubViewer()
@*/
PetscErrorCode  PetscViewerRestoreSubViewer(PetscViewer viewer,MPI_Comm comm,PetscViewer *outviewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);

  if (viewer->ops->restoresubviewer) {
    ierr = (*viewer->ops->restoresubviewer)(viewer,comm,outviewer);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot restore SubViewer PetscViewer for type %s",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}
