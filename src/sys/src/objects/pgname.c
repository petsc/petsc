/*$Id: pgname.c,v 1.24 2001/01/15 21:43:52 bsmith Exp bsmith $*/

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetName"
/*@C
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetName((PetscObject)mat,&name);
-  name - the name associated with obj

   Level: intermediate

   Concepts: object name

.seealso: PetscObjectSetName()
@*/
int PetscObjectGetName(PetscObject obj,char *name[])
{
  int ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  if (!name) SETERRQ(PETSC_ERR_ARG_BADPTR,"Void location for name");
  if (!obj->name) {
    ierr = PetscObjectName(obj);CHKERRQ(ierr);
  }
  *name = obj->name;
  PetscFunctionReturn(0);
}

