/*$Id: pname.c,v 1.35 2000/04/12 04:21:29 bsmith Exp bsmith $*/

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscObjectSetName"></a>*/"PetscObjectSetName"
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
#define __FUNC__ /*<a name="PetscObjectName"></a>*/"PetscObjectName"
/*@C
   PetscObjectName - Gives an object a name if it does not have one

   Not Collective

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject)mat,name);

   Level: advanced

.keywords: object, set, name

.seealso: PetscObjectGetName(), PetscObjectSetName()
@*/
int PetscObjectName(PetscObject obj)
{
  int        ierr;
  char       name[16];
  static int counter = 0;

  PetscFunctionBegin;
  if (!obj->name) {
    sprintf(name,"n_%d",counter++);
    ierr = PetscStrallocpy(name,&obj->name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscObjectPublish"></a>*/"PetscObjectPublish"
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
#define __FUNC__ /*<a name=PetscObjectChangeTypeName""></a>*/"PetscObjectChangeTypeName"
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




