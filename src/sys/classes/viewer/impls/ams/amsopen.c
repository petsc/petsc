
#include <petsc-private/viewerimpl.h>   /*I  "petscsys.h"  */
#include <petscviewerams.h>

#undef __FUNCT__
#define __FUNCT__ "PetscViewerAMSOpen"
/*@C
    PetscViewerAMSOpen - Opens an AMS memory snooper PetscViewer.

    Collective on MPI_Comm

    Input Parameters:
.   comm - the MPI communicator

    Output Parameter:
.   lab - the PetscViewer

    Options Database Keys:
+   -ams_port <port number> - port number where you are running AMS client
.   -xxx_view ams - publish the object xxx
-   -xxx_ams_block - blocks the program at the end of a critical point (for KSP and SNES it is the end of a solve) until
                    the user unblocks the the problem with an external tool that access the object with the AMS

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.


    Notes:
    Unlike other viewers that only access the object being viewed on the call to XXXView(object,viewer) the AMS viewer allows
    one to view the object asynchronously as the program continues to run. One can remove AMS access to the object with a call to
    PetscObjectAMSViewOff().

    Information about the AMS is available via http://www.mcs.anl.gov/ams.

   Concepts: AMS
   Concepts: Argonne Memory Snooper
   Concepts: Asynchronous Memory Snooper

.seealso: PetscViewerDestroy(), PetscViewerStringSPrintf(), PETSC_VIEWER_AMS_(), PetscObjectAMSBlock(),
          PetscObjectAMSViewOff(), PetscObjectAMSTakeAccess(), PetscObjectAMSGrantAccess()

@*/
PetscErrorCode PetscViewerAMSOpen(MPI_Comm comm,PetscViewer *lab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERAMS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectViewAMS"
/*@C
   PetscObjectViewAMS - View the base portion of any object with an AMS viewer

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);
-  viewer - the AMS viewer

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSViewOff()

@*/
PetscErrorCode  PetscObjectViewAMS(PetscObject obj,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->amsmem) PetscFunctionReturn(0);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);

  PetscStackCallAMS(AMS_Memory_Create,(obj->name,&obj->amsmem));
  PetscStackCallAMS(AMS_New_Field,(obj->amsmem,"Class",&obj->class_name,1,AMS_READ,AMS_STRING));
  PetscStackCallAMS(AMS_New_Field,(obj->amsmem,"Type",&obj->type_name,1,AMS_READ,AMS_STRING));
  PetscStackCallAMS(AMS_New_Field,(obj->amsmem,"Id",&obj->id,1,AMS_READ,AMS_INT));
  PetscStackCallAMS(AMS_New_Field,(obj->amsmem,"ParentId",&obj->parentid,1,AMS_READ,AMS_INT));
  PetscStackCallAMS(AMS_New_Field,(obj->amsmem,"Name",&obj->name,1,AMS_READ,AMS_STRING));
  PetscStackCallAMS(AMS_New_Field,(obj->amsmem,"Publish Block",&obj->amspublishblock,1,AMS_READ,AMS_BOOLEAN));
  PetscStackCallAMS(AMS_New_Field,(obj->amsmem,"Block",&obj->amsblock,1,AMS_WRITE,AMS_BOOLEAN));
  PetscFunctionReturn(0);
}
