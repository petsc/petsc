
#include <petsc-private/viewerimpl.h>
#include <petscviewerams.h>
#include <petscsys.h>

/*
    The variable Petsc_Viewer_Ams_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static PetscMPIInt Petsc_Viewer_Ams_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__
#define __FUNCT__ "PETSC_VIEWER_AMS_"
/*@C
     PETSC_VIEWER_AMS_ - Creates an AMS memory snooper PetscViewer shared by all processors
                   in a communicator.

     Collective on MPI_Comm

     Input Parameters:
.    comm - the MPI communicator to share the PetscViewer

     Level: developer

     Notes:
     Unlike almost all other PETSc routines, PETSC_VIEWER_AMS_() does not return
     an error code.  The window PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_AMS_(comm));

.seealso: PETSC_VIEWER_AMS_WORLD, PETSC_VIEWER_AMS_SELF
@*/
PetscViewer PETSC_VIEWER_AMS_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flag;
  PetscViewer    viewer;
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&ncomm,NULL);if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Ams_keyval,0);
    if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,PETSC_ERROR_INITIAL," "); viewer = 0;}
  }
  ierr = MPI_Attr_get(ncomm,Petsc_Viewer_Ams_keyval,(void**)&viewer,&flag);
  if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,PETSC_ERROR_INITIAL," "); viewer = 0;}
  if (!flag) { /* PetscViewer not yet created */
    ierr = PetscViewerAMSOpen(comm,&viewer);CHKERRQ(ierr);
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,PETSC_ERROR_INITIAL," "); viewer = 0;}
    ierr = MPI_Attr_put(ncomm,Petsc_Viewer_Ams_keyval,(void*)viewer);
    if (ierr) {PetscError(ncomm,__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,1,PETSC_ERROR_INITIAL," "); viewer = 0;}
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_AMS_",__FILE__,__SDIR__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  PetscFunctionReturn(viewer);
}

/*
       If there is a PetscViewer associated with this communicator, it is destroyed.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscViewer_AMS_Destroy"
PetscErrorCode PetscViewer_AMS_Destroy(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flag;
  PetscViewer    viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Ams_keyval == MPI_KEYVAL_INVALID) PetscFunctionReturn(0);

  ierr = MPI_Attr_get(comm,Petsc_Viewer_Ams_keyval,(void**)&viewer,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Ams_keyval);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerDestroy_AMS"
static PetscErrorCode PetscViewerDestroy_AMS(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Make sure that we mark that the stack is no longer published
  */
  if (PetscObjectComm((PetscObject)viewer) == PETSC_COMM_WORLD) {
    ierr = PetscStackAMSViewOff();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerCreate_AMS"
PETSC_EXTERN PetscErrorCode PetscViewerCreate_AMS(PetscViewer v)
{
  PetscFunctionBegin;
  v->ops->destroy = PetscViewerDestroy_AMS;
  PetscFunctionReturn(0);
}



