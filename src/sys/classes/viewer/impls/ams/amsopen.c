
#include <petsc-private/viewerimpl.h>   /*I  "petscsys.h"  */
#include <petscviewersaws.h>

#undef __FUNCT__
#define __FUNCT__ "PetscViewerSAWsOpen"
/*@C
    PetscViewerSAWsOpen - Opens an SAWs memory snooper PetscViewer.

    Collective on MPI_Comm

    Input Parameters:
.   comm - the MPI communicator

    Output Parameter:
.   lab - the PetscViewer

    Options Database Keys:
+   -ams_port <port number> - port number where you are running SAWs client
.   -xxx_view ams - publish the object xxx
-   -xxx_saws_block - blocks the program at the end of a critical point (for KSP and SNES it is the end of a solve) until
                    the user unblocks the the problem with an external tool that access the object with the AMS

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.


    Notes:
    Unlike other viewers that only access the object being viewed on the call to XXXView(object,viewer) the SAWs viewer allows
    one to view the object asynchronously as the program continues to run. One can remove SAWs access to the object with a call to
    PetscObjectSAWsViewOff().

    Information about the SAWs is available via http://www.mcs.anl.gov/SAWs.

   Concepts: AMS
   Concepts: Argonne Memory Snooper
   Concepts: Asynchronous Memory Snooper

.seealso: PetscViewerDestroy(), PetscViewerStringSPrintf(), PETSC_VIEWER_SAWS_(), PetscObjectSAWsBlock(),
          PetscObjectSAWsViewOff(), PetscObjectSAWsTakeAccess(), PetscObjectSAWsGrantAccess()

@*/
PetscErrorCode PetscViewerSAWsOpen(MPI_Comm comm,PetscViewer *lab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,lab);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*lab,PETSCVIEWERSAWS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectViewSAWs"
/*@C
   PetscObjectViewSAWs - View the base portion of any object with an SAWs viewer

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);
-  viewer - the SAWs viewer

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectSAWsViewOff()

@*/
PetscErrorCode  PetscObjectViewSAWs(PetscObject obj,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->amsmem) PetscFunctionReturn(0);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);

  PetscStackCallSAWs(SAWS_Directory_Create,(obj->name,&obj->amsmem));
  PetscStackCallSAWs(SAWS_New_Variable,(obj->amsmem,"Class",&obj->class_name,1,SAWS_READ,SAWS_STRING));
  PetscStackCallSAWs(SAWS_New_Variable,(obj->amsmem,"Type",&obj->type_name,1,SAWS_READ,SAWS_STRING));
  PetscStackCallSAWs(SAWS_New_Variable,(obj->amsmem,"Id",&obj->id,1,SAWS_READ,SAWS_INT));
  PetscStackCallSAWs(SAWS_New_Variable,(obj->amsmem,"ParentId",&obj->parentid,1,SAWS_READ,SAWS_INT));
  PetscStackCallSAWs(SAWS_New_Variable,(obj->amsmem,"Name",&obj->name,1,SAWS_READ,SAWS_STRING));
  PetscStackCallSAWs(SAWS_New_Variable,(obj->amsmem,"Publish Block",&obj->amspublishblock,1,SAWS_READ,SAWS_BOOLEAN));
  PetscStackCallSAWs(SAWS_New_Variable,(obj->amsmem,"Block",&obj->amsblock,1,SAWS_WRITE,SAWS_BOOLEAN));
  PetscFunctionReturn(0);
}
