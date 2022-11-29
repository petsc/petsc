
#include <petsc/private/petscimpl.h> /*I    "petscsys.h"   I*/

/*@C
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Not Collective unless object has not been named yet

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (`PetscObject`), for example,
         `PetscObjectGetName`((`PetscObject`)mat,&name);
-  name - the name associated with obj

   Note:
    Calls `PetscObjectName()` if a name has not yet been provided to the object.

   Level: intermediate

.seealso: `PetscObjectSetName()`, `PetscObjectName()`
@*/
PetscErrorCode PetscObjectGetName(PetscObject obj, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidPointer(name, 2);
  PetscCall(PetscObjectName(obj));
  *name = obj->name;
  PetscFunctionReturn(0);
}
