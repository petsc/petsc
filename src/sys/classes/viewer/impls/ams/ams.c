#include <petsc/private/viewerimpl.h>
#include <petscviewersaws.h>
#include <petscsys.h>

/*
    The variable Petsc_Viewer_SAWs_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static PetscMPIInt Petsc_Viewer_SAWs_keyval = MPI_KEYVAL_INVALID;

/*@C
     PETSC_VIEWER_SAWS_ - Creates a SAWs `PetscViewer` shared by all MPI processes in a communicator.

     Collective

     Input Parameter:
.    comm - the MPI communicator to share the `PetscViewer`

     Level: developer

     Note:
     Unlike almost all other PETSc routines, `PETSC_VIEWER_SAWS_()` does not return
     an error code.  The resulting `PetscViewer` is usually used in the form
.vb
     XXXView(XXX object, PETSC_VIEWER_SAWS_(comm));
.ve

.seealso: [](sec_viewers), `PetscViewer`, `PETSC_VIEWER_SAWS_WORLD`, `PETSC_VIEWER_SAWS_SELF`
@*/
PetscViewer PETSC_VIEWER_SAWS_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    iflg;
  PetscViewer    viewer;
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  PetscCallNull(PetscCommDuplicate(comm, &ncomm, NULL));
  if (Petsc_Viewer_SAWs_keyval == MPI_KEYVAL_INVALID) PetscCallMPINull(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &Petsc_Viewer_SAWs_keyval, 0));
  ierr = (PetscErrorCode)MPI_Comm_get_attr(ncomm, Petsc_Viewer_SAWs_keyval, (void **)&viewer, &iflg);
  if (ierr) {
    ierr = PetscError(ncomm, __LINE__, "PETSC_VIEWER_SAWS_", __FILE__, PETSC_ERR_MPI, PETSC_ERROR_INITIAL, " ");
    PetscFunctionReturn(NULL);
  }
  if (!iflg) { /* PetscViewer not yet created */
    PetscCallNull(PetscViewerSAWsOpen(comm, &viewer));
    PetscCallNull(PetscObjectRegisterDestroy((PetscObject)viewer));
    ierr = (PetscErrorCode)MPI_Comm_set_attr(ncomm, Petsc_Viewer_SAWs_keyval, (void *)viewer);
    if (ierr) {
      ierr = PetscError(ncomm, __LINE__, "PETSC_VIEWER_SAWS_", __FILE__, PETSC_ERR_MPI, PETSC_ERROR_INITIAL, " ");
      PetscFunctionReturn(NULL);
    }
  }
  PetscCallNull(PetscCommDestroy(&ncomm));
  PetscFunctionReturn(viewer);
}

static PetscErrorCode PetscViewerDestroy_SAWs(PetscViewer viewer)
{
  PetscFunctionBegin;
  /*
     Make sure that we mark that the stack is no longer published
  */
  if (PetscObjectComm((PetscObject)viewer) == PETSC_COMM_WORLD) PetscCall(PetscStackSAWsViewOff());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscViewerCreate_SAWs(PetscViewer v)
{
  PetscFunctionBegin;
  v->ops->destroy = PetscViewerDestroy_SAWs;
  PetscFunctionReturn(PETSC_SUCCESS);
}
