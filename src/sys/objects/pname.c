#define PETSC_DLL

#include "petscsys.h"        /*I    "petscsys.h"   I*/

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
  PetscValidHeader(obj,1);
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
  PetscErrorCode   ierr;
  PetscCommCounter *counter;
  PetscMPIInt      flg;
  char             name[64];

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (!obj->name) {
    ierr = MPI_Attr_get(obj->comm,Petsc_Counter_keyval,(void*)&counter,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Bad MPI communicator supplied; must be a PETSc communicator");
    ierr = PetscSNPrintf(name,64,"%s_%D",obj->class_name,counter->namecount++);CHKERRQ(ierr);
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
  PetscValidHeader(obj,1);
  ierr = PetscObjectTakeAccess(obj);CHKERRQ(ierr);
  ierr = PetscStrfree(obj->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type_name,&obj->type_name);CHKERRQ(ierr);
  ierr = PetscObjectGrantAccess(obj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

