
#include "petscsys.h"        /*I    "petscsys.h"   I*/

/*
     If true then every PETSc object will be published with the AMS
*/
PetscTruth PetscAMSPublishAll;

/*
    Publishes the common header part of any PETSc object to the AMS
*/
#undef __FUNCT__  
#define __FUNCT__ "PetscObjectPublishBaseBegin"
int PetscObjectPublishBaseBegin(PetscObject obj)
{
  AMS_Memory     amem;
  AMS_Comm       acomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectName(obj);CHKERRQ(ierr);

  ierr      = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(obj->comm),&acomm);CHKERRQ(ierr);
  ierr      = AMS_Memory_create(acomm,obj->name,&amem);CHKERRQ(ierr);
  obj->amem = (int)amem;

  ierr = AMS_Memory_take_access(amem);CHKERRQ(ierr); 
  ierr = AMS_Memory_add_field(amem,"Class",&obj->class_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Type",&obj->type_name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Id",&obj->id,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"ParentId",&obj->parentid,1,AMS_INT,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Name",&obj->name,1,AMS_STRING,AMS_READ,AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscObjectPublishBaseEnd"
int PetscObjectPublishBaseEnd(PetscObject obj)
{
  AMS_Memory     amem = (AMS_Memory) obj->amem;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (amem < 0) SETERRQ(obj->comm,PETSC_ERR_ARG_WRONGSTATE,"Called without a call to PetscObjectPublishBaseBegin()");
  ierr = AMS_Memory_publish(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(amem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectPublishBaseDestroy"
int PetscObjectPublishBaseDestroy(PetscObject obj)
{
  AMS_Comm       acomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr      = PetscViewerAMSGetAMSComm(PETSC_VIEWER_AMS_(obj->comm),&acomm);CHKERRQ(ierr);
  ierr      = AMS_Memory_destroy(obj->amem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

