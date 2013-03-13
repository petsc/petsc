
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

.seealso: PetscObjectSetName(), PetscObjectAMSUnPublish()

@*/
PetscErrorCode  PetscObjectViewAMS(PetscObject obj,PetscViewer viewer)
{
  PetscErrorCode ierr;
  AMS_Memory     amem;
  AMS_Comm       acomm;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (obj->amsmem != -1) PetscFunctionReturn(0);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);

  ierr = PetscViewerAMSGetAMSComm(viewer,&acomm);CHKERRQ(ierr);
  ierr        = AMS_Memory_create(acomm,obj->name,&amem);CHKERRQ(ierr);
  obj->amsmem = (int)amem;

  ierr = AMS_Memory_take_access(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Class",&obj->class_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Type",&obj->type_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Id",&obj->id,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"ParentId",&obj->parentid,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Name",&obj->name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Block",&obj->amspublishblock,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_publish(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(amem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
