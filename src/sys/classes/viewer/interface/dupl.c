#include <petsc/private/viewerimpl.h> /*I "petscviewer.h" I*/

/*@
  PetscViewerGetSubViewer - Creates a new `PetscViewer` (same type as the old)
  that lives on a subcommunicator of the original viewer's communicator

  Collective

  Input Parameters:
+ viewer - the `PetscViewer` to be reproduced
- comm   - the sub communicator to use

  Output Parameter:
. outviewer - new `PetscViewer`

  Level: advanced

  Notes:
  The output of the subviewers is synchronized against the original `viewer`. For example, if a
  viewer on two MPI processes is decomposed into two subviewers, the output from the first viewer is
  all printed before the output from the second viewer.

  Call `PetscViewerRestoreSubViewer()` to destroy this `PetscViewer`, NOT `PetscViewerDestroy()`

  This is most commonly used to view a sequential object that is part of a
  parallel object. For example `PCView()` on a `PCBJACOBI` could use this to obtain a
  `PetscViewer` that is used with the sequential `KSP` on one block of the preconditioner.

  `PetscViewerFlush()` is run automatically at the beginning of `PetscViewerGetSubViewer()` and with `PetscViewerRestoreSubViewer()`
  for `PETSCVIEWERASCII`

  `PETSCVIEWERDRAW` and `PETSCVIEWERBINARY` only support returning a singleton viewer on MPI rank 0,
  all other ranks will return a `NULL` viewer

  Must be called by all MPI processes that share `viewer`, for processes that are not of interest you can pass
  `PETSC_COMM_SELF`.

  For `PETSCVIEWERASCII` the viewers behavior is as follows\:
.vb
  Recursive calls are allowed
  A call to `PetscViewerASCIIPrintf()` on a subviewer results in output for the first MPI process in the `outviewer` only
  Calls to  `PetscViewerASCIIPrintf()` and `PetscViewerASCIISynchronizedPrintf()` are immediately passed up through all
  the parent viewers to the higher most parent with `PetscViewerASCIISynchronizedPrintf()` where they are immediately
  printed on the first MPI process or stashed on the other processes.
  At the higher most `PetscViewerRestoreSubViewer()` the viewer is automatically flushed with `PetscViewerFlush()`
.ve

  Developer Notes:
  There is currently incomplete error checking to ensure the user does not use the original viewer between the
  the calls to `PetscViewerGetSubViewer()` and `PetscViewerRestoreSubViewer()`. If the user does there
  could be errors in the viewing that go undetected or crash the code.

  Complex use of this functionality with `PETSCVIEWERASCII` can result in output in unexpected order. This seems unavoidable.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerSocketOpen()`, `PetscViewerASCIIOpen()`, `PetscViewerDrawOpen()`,
          `PetscViewerFlush()`, `PetscViewerRestoreSubViewer()`
@*/
PetscErrorCode PetscViewerGetSubViewer(PetscViewer viewer, MPI_Comm comm, PetscViewer *outviewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(outviewer, 3);
  PetscUseTypeMethod(viewer, getsubviewer, comm, outviewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerRestoreSubViewer - Restores a  `PetscViewer` obtained with `PetscViewerGetSubViewer()`.

  Collective

  Input Parameters:
+ viewer    - the `PetscViewer` that was reproduced
. comm      - the sub communicator
- outviewer - the subviewer to be returned

  Level: advanced

  Notes:
  Automatically runs `PetscViewerFlush()` on `outviewer`

  Must be called by all MPI processes that share `viewer`, for processes that are not of interest you can pass
  `PETSC_COMM_SELF`.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerSocketOpen()`, `PetscViewerASCIIOpen()`, `PetscViewerDrawOpen()`, `PetscViewerGetSubViewer()`,
          `PetscViewerFlush()`
@*/
PetscErrorCode PetscViewerRestoreSubViewer(PetscViewer viewer, MPI_Comm comm, PetscViewer *outviewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);

  PetscUseTypeMethod(viewer, restoresubviewer, comm, outviewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}
