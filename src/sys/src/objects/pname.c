#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pname.c,v 1.16 1998/08/26 22:01:46 balay Exp bsmith $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetName"
/*@C 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject) mat,name);
-  name - the name to give obj

.keywords: object, set, name

.seealso: PetscObjectGetName()
@*/
int PetscObjectSetName(PetscObject obj,const char name[])
{
  int len;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  len = PetscStrlen(name);
  obj->name = (char *)PetscMalloc(sizeof(char)*(len+1)); CHKPTRQ(obj->name);
  PetscStrcpy(obj->name,name);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectPublish"
/*@C 
   PetscObjectPublish - Publishs an object for the ALICE Memory Snooper

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject) mat,name);

.keywords: object, monitoring, publishing

.seealso: PetscObjectSetName()
@*/
int PetscObjectPublish(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  if (obj->bops->publish) {
    ierr = (*obj->bops->publish)(obj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    Publishes the common header part of any PETSc object. 
 */
#undef __FUNC__  
#define __FUNC__ "PetscObjectPublishHeaderBegin"
int PetscObjectPublishBaseBegin(PetscObject obj,char *memname)
{
#if defined(HAVE_AMS)
  AMS_Memory amem;
  AMS_Comm   acomm;
  int        ierr;

  PetscFunctionBegin;
  ierr      = ViewerAMSGetAMSComm(VIEWER_AMS_(obj->comm),&acomm);CHKERRQ(ierr);
  ierr      = AMS_Memory_create(acomm,memname,&amem);CHKERRQ(ierr);
  obj->amem = (int) amem;

  ierr = AMS_Memory_take_access(amem);CHKERRQ(ierr); 
  ierr = AMS_Memory_add_field(amem,"Type",&obj->type_name,1,AMS_STRING,AMS_READ,
                                AMS_COMMON,AMS_REDUCT_UNDEF);CHKERRQ(ierr);

#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectPublishHeaderEnd"
int PetscObjectPublishBaseEnd(PetscObject obj)
{
#if defined(HAVE_AMS)
  AMS_Memory amem = (AMS_Memory) obj->amem;
  int        ierr;

  PetscFunctionBegin;

  if (amem < 0) SETERRQ(1,1,"Called without a call to PetscObjectPublishBaseBegin()");
  ierr = AMS_Memory_publish(amem);CHKERRQ(ierr);
  ierr = AMS_Memory_grant_access(amem);CHKERRQ(ierr);

#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

