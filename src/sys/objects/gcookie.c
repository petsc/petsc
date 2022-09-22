
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@
   PetscObjectGetClassId - Gets the classid for any `PetscObject`

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
         Thus must be cast with a (`PetscObject`), for example,
         `PetscObjectGetClassId`((`PetscObject`)mat,&classid);

   Output Parameter:
.  classid - the classid

   Level: developer

.seealso: `PetscObject`, `PetscClassId`, `PetscObjectGetClassName()`, `PetscObjectGetType()`
@*/
PetscErrorCode PetscObjectGetClassId(PetscObject obj, PetscClassId *classid)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidIntPointer(classid, 2);
  *classid = obj->classid;
  PetscFunctionReturn(0);
}

/*@C
   PetscObjectGetClassName - Gets the class name for any `PetscObject`

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`.
         Thus must be cast with a (`PetscObject`), for example,
         `PetscObjectGetClassName`((`PetscObject`)mat,&classname);

   Output Parameter:
.  classname - the class name, for example "Vec"

   Level: developer

.seealso: `PetscObject`, `PetscClassId`, `PetscObjectGetType()`
@*/
PetscErrorCode PetscObjectGetClassName(PetscObject obj, const char *classname[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscValidPointer(classname, 2);
  *classname = obj->class_name;
  PetscFunctionReturn(0);
}
