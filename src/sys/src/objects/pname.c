/*$Id: pname.c,v 1.42 2001/04/10 19:34:33 bsmith Exp $*/

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
int PetscObjectSetName(PetscObject obj,const char name[])
{
  int ierr;

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
int PetscObjectName(PetscObject obj)
{
  int        ierr;
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
   PetscObjectPublish - Publishs an object for the ALICE Memory Snooper

   Collective on PetscObject

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

   Concepts: publishing object
   Concepts: AMS
   Concepts: ALICE Memory Snooper
   Concepts: Asynchronous Memory Snooper

.seealso: PetscObjectSetName(), PetscViewerAMSOpen()

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

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectChangeTypeName"
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

#undef __FUNCT__
#define __FUNCT__ "PetscObjectChangeSerializeName"
/*@C
  PetscObjectChangeSerializeName - Changes the serializer name.

  Not Collective

  Input Parameters:
+ obj            - The PETSc object, for example a Vec, Mat or KSP.
- serialize_name - The string containing a serializer name

  Note:
  This works for any PETSc object, and thus must be cast with a (PetscObject).

  Level: intermediate

.keywords: changing serializers
.seealso: PetscObjectChangeTypeName()
@*/
int PetscObjectChangeSerializeName(PetscObject obj, char *serialize_name)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTakeAccess(obj);                                                                      CHKERRQ(ierr);
  ierr = PetscStrfree(obj->serialize_name);                                                               CHKERRQ(ierr);
  ierr = PetscStrallocpy(serialize_name, &obj->serialize_name);                                           CHKERRQ(ierr);
  ierr = PetscObjectGrantAccess(obj);                                                                     CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
