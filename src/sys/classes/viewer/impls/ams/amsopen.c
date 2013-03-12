
#include <petsc-private/viewerimpl.h>   /*I  "petscsys.h"  */
#include <petscviewerams.h>

#undef __FUNCT__
#define __FUNCT__ "PetscViewerAMSOpen"
/*@C
    PetscViewerAMSOpen - Opens an AMS memory snooper PetscViewer.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the MPI communicator
-   name - name of AMS communicator being created

    Output Parameter:
.   lab - the PetscViewer

    Options Database Keys:
+   -ams_port <port number> - port number where you are running AMS client
.   -ams_publish_objects - publish all PETSc objects to be visible to the AMS memory snooper,
                           use PetscObjectAMSPublish() to publish individual objects
-   -ams_java - open JAVA AMS client

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

    See the matlab/petsc directory in the AMS installation for one example of external
    tools that can monitor PETSc objects that have been published.

    Notes:
    This PetscViewer can be destroyed with PetscViewerDestroy().

    This viewer is currently different than other viewers in that you cannot pass this viewer to XXXView() to view the XXX object.
    PETSC_VIEWER_AMS_() is used by PetscObjectAMSPublish() to connect to that particular AMS communicator.

    Information about the AMS is available via http://www.mcs.anl.gov/ams.

   Concepts: AMS
   Concepts: ALICE Memory Snooper
   Concepts: Asynchronous Memory Snooper

.seealso: PetscObjectAMSPublish(), PetscViewerDestroy(), PetscViewerStringSPrintf(), PETSC_VIEWER_AMS_(),
          PetscObjectAMSPublish(), PetscObjectAMSUnPublish(), PetscObjectAMSTakeAccess(), PetscObjectAMSGrantAccess()

@*/
PetscErrorCode PetscViewerAMSOpen(MPI_Comm comm,const char name[],PetscViewer *lab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERAMS);CHKERRQ(ierr);
  ierr = PetscViewerAMSSetCommName(*lab,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
