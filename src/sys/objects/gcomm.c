/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h> /*I   "petscsys.h"    I*/

/*@C
  PetscObjectComm - Gets the MPI communicator for any `PetscObject` regardless of the type.

  Not Collective, No Fortran Support

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be
        cast with a (`PetscObject`), for example, `PetscObjectComm`((`PetscObject`)mat,...);

  Level: advanced

  Note:
  Returns the MPI communicator or `MPI_COMM_NULL` if `obj` is not valid.

  This is one of the rare PETSc routines that does not return an error code. Use `PetscObjectGetComm()`
  when appropriate for error handling.

.seealso: `PetscObject`, `PetscObjectGetComm()`
@*/
MPI_Comm PetscObjectComm(PetscObject obj)
{
  return obj ? obj->comm : MPI_COMM_NULL;
}

/*@C
  PetscObjectGetComm - Gets the MPI communicator for any `PetscObject` regardless of the type.

  Not Collective

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
        `PetscObjectGetComm`((`PetscObject`)mat,&comm);

  Output Parameter:
. comm - the MPI communicator

  Level: advanced

.seealso: `PetscObject`, `PetscObjectComm()`
@*/
PetscErrorCode PetscObjectGetComm(PetscObject obj, MPI_Comm *comm)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(comm, 2);
  *comm = obj->comm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectGetTabLevel - Gets the number of tabs that `PETSCVIEWERASCII` output for that object uses

  Not Collective

  Input Parameter:
. obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
        `PetscObjectGetTabLevel`((`PetscObject`)mat,&tab);

  Output Parameter:
. tab - the number of tabs

  Level: developer

  Note:
  This is used to manage the output from options that are embedded in other objects. For example
  the `KSP` object inside a `SNES` object. By indenting each lower level further the hierarchy of objects
  is clear.

.seealso: `PetscObjectIncrementTabLevel()`, `PetscObjectSetTabLevel()`, `PETSCVIEWERASCII`, `PetscObject`
@*/
PetscErrorCode PetscObjectGetTabLevel(PetscObject obj, PetscInt *tab)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  PetscAssertPointer(tab, 2);
  *tab = obj->tablevel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectSetTabLevel - Sets the number of tabs that `PETSCVIEWERASCII` output for that object uses

  Not Collective

  Input Parameters:
+ obj - any PETSc object, for example a `Vec`, `Mat` or `KSP`. It must be cast with a (`PetscObject`), for example,
        `PetscObjectSetTabLevel`((`PetscObject`)mat,tab;
- tab - the number of tabs

  Level: developer

  Notes:
  this is used to manage the output from options that are embedded in other objects. For example
  the `KSP` object inside a `SNES` object. By indenting each lower level further the hierarchy of objects
  is clear.

  `PetscObjectIncrementTabLevel()` is the preferred API

.seealso: `PetscObjectIncrementTabLevel()`, `PetscObjectGetTabLevel()`
@*/
PetscErrorCode PetscObjectSetTabLevel(PetscObject obj, PetscInt tab)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  obj->tablevel = tab;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscObjectIncrementTabLevel - Increments the number of tabs that `PETSCVIEWERASCII` output for that object use based on
  the tablevel of another object. This should be called immediately after the object is created.

  Not Collective

  Input Parameters:
+ obj    - any PETSc object where we are changing the tab
. oldobj - the object providing the tab, optional pass `NULL` to use 0 as the previous tablevel for `obj`
- tab    - the increment that is added to the old objects tab

  Level: developer

  Note:
  this is used to manage the output from options that are embedded in other objects. For example
  the `KSP` object inside a `SNES` object. By indenting each lower level further the hierarchy of objects
  is clear.

.seealso: `PETSCVIEWERASCII`, `PetscObjectSetTabLevel()`, `PetscObjectGetTabLevel()`
@*/
PetscErrorCode PetscObjectIncrementTabLevel(PetscObject obj, PetscObject oldobj, PetscInt tab)
{
  PetscFunctionBegin;
  PetscValidHeader(obj, 1);
  if (oldobj) PetscValidHeader(oldobj, 2);
  obj->tablevel = (oldobj ? oldobj->tablevel : 0) + tab;
  PetscFunctionReturn(PETSC_SUCCESS);
}
