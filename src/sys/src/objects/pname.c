#define PETSC_DLL

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectSetName"
/*@C 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject)mat,name);
-  name - the name to give obj

   Level: advanced

   Concepts: object name^setting

.seealso: PetscObjectGetName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectSetName(PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  ierr = PetscStrfree(obj->name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&obj->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectName"
/*@C
   PetscObjectName - Gives an object a name if it does not have one

   Not Collective

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

   Concepts: object name^setting default

.seealso: PetscObjectGetName(), PetscObjectSetName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectName(PetscObject obj)
{
  PetscErrorCode ierr;
  char       name[64];
  static int counter = 0;

  PetscFunctionBegin;
  if (!obj->name) {
    sprintf(name,"%s_%d",obj->class_name,counter++);
    ierr = PetscStrallocpy(name,&obj->name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectPublish"
/*@C 
   PetscObjectPublish - Publish an object

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

   Concepts: publishing object

   Notes: Not currently used

.seealso: PetscObjectSetName()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectPublish(PetscObject obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (obj->bops->publish) {
    ierr = (*obj->bops->publish)(obj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectChangeTypeName"
PetscErrorCode PETSC_DLLEXPORT PetscObjectChangeTypeName(PetscObject obj,const char type_name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTakeAccess(obj);CHKERRQ(ierr);
  ierr = PetscStrfree(obj->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type_name,&obj->type_name);CHKERRQ(ierr);
  ierr = PetscObjectGrantAccess(obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

