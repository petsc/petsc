/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@
  PetscObjectGetType - Gets the object type of any `PetscObject`.

  Not Collective

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
        `PetscObjectGetType`((`PetscObject`)mat,&type);

  Output Parameter:
. type - the object type, for example, `MATSEQAIJ`

  Level: advanced

.seealso: `PetscObject`, `PetscClassId`, `PetscObjectGetClassName()`, `PetscObjectGetClassId()`
@*/
PetscErrorCode PetscObjectGetType(PetscObject obj, const char *type[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(type, 2);
  *type = obj->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}
