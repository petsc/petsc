#define PETSC_DLL

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetName"
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
PetscErrorCode PETSC_DLLEXPORT PetscObjectGetName(PetscObject obj,char *name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  if (!name) SETERRQ(PETSC_ERR_ARG_BADPTR,"Void location for name");
  if (!obj->name) {
    ierr = PetscObjectName(obj);CHKERRQ(ierr);
  }
  *name = obj->name;
  PetscFunctionReturn(0);
}

