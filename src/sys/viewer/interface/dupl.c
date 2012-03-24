
#include <petsc-private/viewerimpl.h>  /*I "petscviewer.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerGetSingleton" 
/*@
   PetscViewerGetSingleton - Creates a new PetscViewer (same type as the old)
    that lives on a single processor (with MPI_comm PETSC_COMM_SELF)

    Collective on PetscViewer

   Input Parameter:
.  viewer - the PetscViewer to be duplicated

   Output Parameter:
.  outviewer - new PetscViewer

   Level: advanced

   Notes: Call PetscViewerRestoreSingleton() to return this PetscViewer, NOT PetscViewerDestroy()

     This is most commonly used to view a sequential object that is part of a 
    parallel object. For example block Jacobi PC view could use this to obtain a
    PetscViewer that is used with the sequential KSP on one block of the preconditioner.

   Concepts: PetscViewer^sequential version

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerRestoreSingleton()
@*/
PetscErrorCode  PetscViewerGetSingleton(PetscViewer viewer,PetscViewer *outviewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(outviewer,2);
  ierr = MPI_Comm_size(((PetscObject)viewer)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
    *outviewer = viewer;
  } else if (viewer->ops->getsingleton) {
    ierr = (*viewer->ops->getsingleton)(viewer,outviewer);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get singleton PetscViewer for type %s",((PetscObject)viewer)->type_name);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerRestoreSingleton" 
/*@
   PetscViewerRestoreSingleton - Restores a new PetscViewer obtained with PetscViewerGetSingleton().

    Collective on PetscViewer

   Input Parameters:
+  viewer - the PetscViewer to be duplicated
-  outviewer - new PetscViewer

   Level: advanced

   Notes: Call PetscViewerGetSingleton() to get this PetscViewer, NOT PetscViewerCreate()

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerGetSingleton()
@*/
PetscErrorCode  PetscViewerRestoreSingleton(PetscViewer viewer,PetscViewer *outviewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);

  ierr = MPI_Comm_size(((PetscObject)viewer)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    if (outviewer) *outviewer = 0;
  } else if (viewer->ops->restoresingleton) {
    ierr = (*viewer->ops->restoresingleton)(viewer,outviewer);CHKERRQ(ierr);
  } 
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerGetSubcomm" 
/*@
   PetscViewerGetSubcomm - Creates a new PetscViewer (same type as the old)
    that lives on a subgroup of processors 

    Collective on PetscViewer

   Input Parameter:
+  viewer - the PetscViewer to be duplicated
-  subcomm - MPI communicator

   Output Parameter:
.  outviewer - new PetscViewer

   Level: advanced

   Notes: Call PetscViewerRestoreSubcomm() to return this PetscViewer, NOT PetscViewerDestroy()

     This is used to view a sequential or a parallel object that is part of a larger
    parallel object. For example redundant PC view could use this to obtain a
    PetscViewer that is used within a subcommunicator on one duplicated preconditioner.

   Concepts: PetscViewer^sequential version

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerRestoreSubcomm()
@*/
PetscErrorCode  PetscViewerGetSubcomm(PetscViewer viewer,MPI_Comm subcomm,PetscViewer *outviewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(outviewer,3);
  ierr = MPI_Comm_size(((PetscObject)viewer)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
    *outviewer = viewer;
  } else if (viewer->ops->getsubcomm) {
    ierr = (*viewer->ops->getsubcomm)(viewer,subcomm,outviewer);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get subcommunicator PetscViewer for type %s",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerRestoreSubcomm" 
/*@
   PetscViewerRestoreSubcomm - Restores a new PetscViewer obtained with PetscViewerGetSubcomm().

    Collective on PetscViewer

   Input Parameters:
+  viewer - the PetscViewer to be duplicated
.  subcomm - MPI communicator
-  outviewer - new PetscViewer

   Level: advanced

   Notes: Call PetscViewerGetSubcomm() to get this PetscViewer, NOT PetscViewerCreate()

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerGetSubcomm()
@*/
PetscErrorCode  PetscViewerRestoreSubcomm(PetscViewer viewer,MPI_Comm subcomm,PetscViewer *outviewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);

  ierr = MPI_Comm_size(((PetscObject)viewer)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    if (outviewer) *outviewer = 0;
  } else if (viewer->ops->restoresubcomm) {
    ierr = (*viewer->ops->restoresubcomm)(viewer,subcomm,outviewer);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

