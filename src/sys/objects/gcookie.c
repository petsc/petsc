/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@
  PetscObjectGetClassId - Gets the classid for any `PetscObject`

  Not Collective

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
        `PetscObjectGetClassId`((`PetscObject`)mat,&classid);

  Output Parameter:
. classid - the classid

  Level: developer

.seealso: `PetscObject`, `PetscClassId`, `PetscObjectGetClassName()`, `PetscObjectGetType()`
@*/
PetscErrorCode PetscObjectGetClassId(PetscObject obj, PetscClassId *classid)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(classid, 2);
  *classid = obj->classid;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectGetClassName - Gets the class name for any `PetscObject`

  Not Collective

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
        `PetscObjectGetClassName`((`PetscObject`)mat,&classname);

  Output Parameter:
. classname - the class name, for example "Vec"

  Level: developer

.seealso: `PetscObject`, `PetscClassId`, `PetscObjectGetType()`, `PetscObjectGetClassId()`
@*/
PetscErrorCode PetscObjectGetClassName(PetscObject obj, const char *classname[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(classname, 2);
  *classname = obj->class_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}
