/*$Id: pname.c,v 1.32 1999/11/24 21:53:05 bsmith Exp bsmith $*/

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetName"
/*@C 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject)mat,name);
-  name - the name to give obj

   Level: advanced

.keywords: object, set, name

.seealso: PetscObjectGetName()
@*/
int PetscObjectSetName(PetscObject obj,const char name[])
{
  int ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  ierr = PetscStrfree(obj->name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&obj->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectPublish"
/*@C 
   PetscObjectPublish - Publishs an object for the ALICE Memory Snooper

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

.keywords: object, monitoring, publishing

.seealso: PetscObjectSetName(), ViewerAMSOpen()

@*/
int PetscObjectPublish(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj);
  if (obj->bops->publish) {
    ierr = (*obj->bops->publish)(obj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectChangeTypeName"
int PetscObjectChangeTypeName(PetscObject obj,char *type_name)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTakeAccess(obj);CHKERRQ(ierr);
  ierr = PetscStrfree(obj->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type_name,&obj->type_name);CHKERRQ(ierr);
  ierr = PetscObjectGrantAccess(obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




