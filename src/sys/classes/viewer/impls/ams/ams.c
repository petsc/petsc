
#include <petsc/private/viewerimpl.h>
#include <petscviewersaws.h>
#include <petscsys.h>

/*
    The variable Petsc_Viewer_SAWs_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static PetscMPIInt Petsc_Viewer_SAWs_keyval = MPI_KEYVAL_INVALID;

/*@C
     PETSC_VIEWER_SAWS_ - Creates an SAWs PetscViewer shared by all processors in a communicator.

     Collective on MPI_Comm

     Input Parameters:
.    comm - the MPI communicator to share the PetscViewer

     Level: developer

     Notes:
     Unlike almost all other PETSc routines, PETSC_VIEWER_SAWS_() does not return
     an error code.  The resulting PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_SAWS_(comm));

.seealso: PETSC_VIEWER_SAWS_WORLD, PETSC_VIEWER_SAWS_SELF
@*/
PetscViewer PETSC_VIEWER_SAWS_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flag;
  PetscViewer    viewer;
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&ncomm,NULL);if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SAWS_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  if (Petsc_Viewer_SAWs_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_SAWs_keyval,0);
    if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_SAWS_",__FILE__,1,PETSC_ERROR_INITIAL," "); viewer = NULL;}
  }
  ierr = MPI_Attr_get(ncomm,Petsc_Viewer_SAWs_keyval,(void**)&viewer,&flag);
  if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_SAWS_",__FILE__,1,PETSC_ERROR_INITIAL," "); viewer = NULL;}
  if (!flag) { /* PetscViewer not yet created */
    ierr = PetscViewerSAWsOpen(comm,&viewer);
    if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_SAWS_",__FILE__,1,PETSC_ERROR_INITIAL," "); viewer = NULL;}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_SAWS_",__FILE__,1,PETSC_ERROR_INITIAL," "); viewer = NULL;}
    ierr = MPI_Attr_put(ncomm,Petsc_Viewer_SAWs_keyval,(void*)viewer);
    if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_SAWS_",__FILE__,1,PETSC_ERROR_INITIAL," "); viewer = NULL;}
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_SAWS_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  PetscFunctionReturn(viewer);
}

/*
       If there is a PetscViewer associated with this communicator, it is destroyed.
*/
PetscErrorCode PetscViewer_SAWS_Destroy(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flag;
  PetscViewer    viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_SAWs_keyval == MPI_KEYVAL_INVALID) PetscFunctionReturn(0);

  ierr = MPI_Attr_get(comm,Petsc_Viewer_SAWs_keyval,(void**)&viewer,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_SAWs_keyval);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerDestroy_SAWs(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Make sure that we mark that the stack is no longer published
  */
  if (PetscObjectComm((PetscObject)viewer) == PETSC_COMM_WORLD) {
    ierr = PetscStackSAWsViewOff();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscViewerCreate_SAWs(PetscViewer v)
{
  PetscFunctionBegin;
  v->ops->destroy = PetscViewerDestroy_SAWs;
  PetscFunctionReturn(0);
}



