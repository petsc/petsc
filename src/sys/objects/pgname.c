#include <petsc/private/petscimpl.h> /*I    "petscsys.h"   I*/

/*@
  PetscObjectGetName - Gets a string name associated with a PETSc object.

  Not Collective unless `obj` has not yet been named

  Input Parameters:
+ obj  - the PETSc variable. It must be cast with a (`PetscObject`), for example,
         `PetscObjectGetName`((`PetscObject`)mat,&name);
- name - the name associated with `obj`, do not free

  Level: intermediate

  Note:
  Calls `PetscObjectName()` if a name has not yet been provided to the object.

.seealso: `PetscObjectSetName()`, `PetscObjectName()`, `PetscObject`, `PetscObjectGetId()`
@*/
PetscErrorCode PetscObjectGetName(PetscObject obj, const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(name, 2);
  PetscCall(PetscObjectName(obj));
  *name = obj->name;
  PetscFunctionReturn(PETSC_SUCCESS);
}
