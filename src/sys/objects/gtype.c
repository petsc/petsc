
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@C
   PetscObjectGetType - Gets the object type of any `PetscObject`.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
         Thus must be cast with a (`PetscObject`), for example,
         `PetscObjectGetType`((`PetscObject`)mat,&type);

   Output Parameter:
.  type - the object type, for example, `MATSEQAIJ`

   Level: advanced

.seealso: `PetscObject`, `PetscClassId`, `PetscObjectGetClassName()`
@*/
PetscErrorCode PetscObjectGetType(PetscObject obj, const char *type[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidPointer(type, 2);
  *type = obj->type_name;
  PetscFunctionReturn(0);
}
