/*$Id: pams.c,v 1.3 2000/04/09 04:34:45 bsmith Exp bsmith $*/

#include "petsc.h"        /*I    "petsc.h"   I*/


/*
    Publishes the common header part of any PETSc object. 
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectPublishBaseBegin"
int PetscObjectPublishBaseBegin(PetscObject obj)
{
  AMS_Memory amem;
  AMS_Comm   acomm;
  int        ierr;
  static int counter = 0;
  char       name[16];

  PetscFunctionBegin;

  if (obj->name) {
    ierr = PetscStrncpy(name,obj->name,16);CHKERRQ(ierr);
  } else {
    sprintf(name,"n_%d",counter++);
  }

  ierr      = ViewerAMSGetAMSComm(VIEWER_AMS_(obj->comm),&acomm);CHKERRQ(ierr);
  ierr      = AMS_Memory_create(acomm,name,&amem);CHKERRQ(ierr);
  obj->amem = (int)amem;

  ierr = AMS_Memory_take_access(amem);CHKERRQ(ierr); 
  ierr = AMS_Memory_add_field(amem,"Class",&obj->class_name,1,AMS_STRING,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Type",&obj->type_name,1,AMS_STRING,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Id",&obj->id,1,AMS_INT,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"ParentId",&obj->parentid,1,AMS_INT,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  ierr = AMS_Memory_add_field(amem,"Name",&obj->name,1,AMS_STRING,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectPublishBaseEnd"
int PetscObjectPublishBaseEnd(PetscObject obj)
{
  AMS_Memory amem = (AMS_Memory) obj->amem;
  int        ierr;

  PetscFunctionBegin;

  if (amem < 0) SETERRQ(1,1,"Called without a call to PetscObjectPublishBaseBegin()");
  ierr = AMS_Memory_publish(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(amem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





