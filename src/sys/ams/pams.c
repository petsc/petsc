
#include <petsc-private/petscimpl.h>        /*I    "petscsys.h"   I*/
#include <petscviewerams.h>
#include <petscsys.h>

/*
     If true then every PETSc object will be published with the AMS
*/
PetscBool PetscAMSPublishAll;

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSTakeAccess"
/*@C
   PetscObjectAMSTakeAccess - Take access of the data fields that have been published to AMS so they may be changed locally

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSUnPublish(), PetscObjectAMSGrantAccess()

@*/
PetscErrorCode  PetscObjectAMSTakeAccess(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (obj->amsmem != -1) {
    ierr = AMS_Memory_take_access(obj->amsmem);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSGrantAccess"
/*@C
   PetscObjectAMSGrantAccess - Grants access of the data fields that have been published to AMS to the memory snooper to change

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSUnPublish(), PetscObjectAMSTakeAccess()

@*/
PetscErrorCode  PetscObjectAMSGrantAccess(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (obj->amsmem != -1) {
    ierr = AMS_Memory_grant_access(obj->amsmem);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSPublish"
/*@C
   PetscObjectAMSPublish - Publish an object

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);

   Notes: PetscViewer objects are not published

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSUnPublish()

@*/
PetscErrorCode  PetscObjectAMSPublish(PetscObject obj)
{
  PetscErrorCode ierr;
  AMS_Memory     amem;
  AMS_Comm       acomm;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (obj->amsmem != -1) PetscFunctionReturn(0);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);

  ierr = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(PETSC_COMM_WORLD),&acomm);CHKERRQ(ierr);
  /* Really want to attach to correct communicator but then browser needs to access multiple communicators
  ierr      = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(obj->comm),&acomm);CHKERRQ(ierr); */

  ierr        = AMS_Memory_create(acomm,obj->name,&amem);CHKERRQ(ierr);
  obj->amsmem = (int)amem;

  ierr = AMS_Memory_take_access(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Class",&obj->class_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Type",&obj->type_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Id",&obj->id,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"ParentId",&obj->parentid,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Name",&obj->name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Block",&obj->amspublishblock,1,AMS_BOOLEAN,AMS_WRITE,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  if (obj->bops->publish) {
    ierr = (*obj->bops->publish)(obj);CHKERRQ(ierr);
  }
  ierr = AMS_Memory_publish(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(amem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSBlock"
/*@C
   PetscObjectAMSBlock - Blocks the object if PetscObjectAMSSetBlock() has been called

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);


   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSUnPublish(), PetscObjectAMSSetBlock()

@*/
PetscErrorCode  PetscObjectAMSBlock(PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);

  if (!obj->amspublishblock) PetscFunctionReturn(0);
  /* Eventually this will be fixed to check if the AMS client has changed the lock */
  while (1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSSetBlock"
/*@C
   PetscObjectAMSSetBlock - Sets whether an object will block at PetscObjectAMSBlock()

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example,
         PetscObjectSetName((PetscObject)mat,name);
-  flg - whether it should block

   Level: advanced

   Concepts: publishing object

.seealso: PetscObjectSetName(), PetscObjectAMSUnPublish(), PetscObjectAMSBlock()

@*/
PetscErrorCode  PetscObjectAMSSetBlock(PetscObject obj,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->amspublishblock = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectAMSUnPublish"
PetscErrorCode PetscObjectAMSUnPublish(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (obj->amsmem == -1) PetscFunctionReturn(0);
  if (obj->bops->unpublish) {
    ierr = (*obj->bops->unpublish)(obj);CHKERRQ(ierr);
  } else {
    ierr        = AMS_Memory_destroy(obj->amsmem);CHKERRQ(ierr);
    obj->amsmem = -1;
  }
  PetscFunctionReturn(0);
}

