
#include <petscsys.h>        /*I    "petscsys.h"   I*/

/*
     If true then every PETSc object will be published with the AMS
*/
PetscBool  PetscAMSPublishAll;

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

.seealso: PetscObjectSetName(), PetscObjectUnPublish()

@*/
PetscErrorCode  PetscObjectAMSPublish(PetscObject obj)
{
  PetscErrorCode ierr;
  AMS_Memory     amem;
  AMS_Comm       acomm;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (obj->amem != -1) PetscFunctionReturn(0);
  ierr = PetscObjectName(obj);CHKERRQ(ierr);

  ierr      = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(PETSC_COMM_WORLD),&acomm);CHKERRQ(ierr);
  /* Really want to attach to correct communicator but then browser needs to access multiple communicators
  ierr      = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(obj->comm),&acomm);CHKERRQ(ierr); */

  ierr      = AMS_Memory_create(acomm,obj->name,&amem);CHKERRQ(ierr);
  obj->amem = (int)amem;

  ierr = AMS_Memory_take_access(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Class",&obj->class_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Type",&obj->type_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Id",&obj->id,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"ParentId",&obj->parentid,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Name",&obj->name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  if (obj->bops->publish) {
    ierr = (*obj->bops->publish)(obj);CHKERRQ(ierr);
  }
  ierr = AMS_Memory_publish(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(amem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectUnPublish"
PetscErrorCode PetscObjectUnPublish(PetscObject obj)
{
  AMS_Comm       acomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (obj->classid == PETSC_VIEWER_CLASSID) PetscFunctionReturn(0);
  if (obj->amem == -1) PetscFunctionReturn(0);
  ierr      = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(obj->comm),&acomm);CHKERRQ(ierr);
  ierr      = AMS_Memory_destroy(obj->amem);CHKERRQ(ierr);
  obj->amem = -1;
  PetscFunctionReturn(0);
}

