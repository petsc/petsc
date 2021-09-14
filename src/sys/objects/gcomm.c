
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petsc/private/petscimpl.h>  /*I   "petscsys.h"    I*/

/*@C
   PetscObjectComm - Gets the MPI communicator for any PetscObject   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example,
         SETERRQ(PetscObjectComm((PetscObject)mat,...);

   Output Parameter:
.  comm - the MPI communicator or MPI_COMM_NULL if object is not valid

   Level: advanced

   Notes:
    Never use this in the form
$       comm = PetscObjectComm((PetscObject)obj);
        instead use PetscObjectGetComm()

.seealso: PetscObjectGetComm()
@*/
MPI_Comm  PetscObjectComm(PetscObject obj)
{
  if (!obj) return MPI_COMM_NULL;
  return obj->comm;
}

/*@C
   PetscObjectGetComm - Gets the MPI communicator for any PetscObject,
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example,
         PetscObjectGetComm((PetscObject)mat,&comm);

   Output Parameter:
.  comm - the MPI communicator

   Level: advanced

.seealso: PetscObjectComm()
@*/
PetscErrorCode  PetscObjectGetComm(PetscObject obj,MPI_Comm *comm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidPointer(comm,2);
  if (obj->bops->getcomm) {
    ierr = obj->bops->getcomm(obj,comm);CHKERRQ(ierr);
  } else *comm = obj->comm;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectGetTabLevel - Gets the number of tabs that ASCII output for that object use

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example,
         PetscObjectGetComm((PetscObject)mat,&comm);

   Output Parameter:
.   tab - the number of tabs

   Level: developer

    Notes:
    this is used to manage the output from options that are imbedded in other objects. For example
      the KSP object inside a SNES object. By indenting each lower level further the hierarchy of objects
      is very clear.

.seealso:  PetscObjectIncrementTabLevel()

@*/
PetscErrorCode  PetscObjectGetTabLevel(PetscObject obj,PetscInt *tab)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  *tab = obj->tablevel;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectSetTabLevel - Sets the number of tabs that ASCII output for that object use

   Not Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example,
         PetscObjectGetComm((PetscObject)mat,&comm);
-   tab - the number of tabs

   Level: developer

    Notes:
    this is used to manage the output from options that are imbedded in other objects. For example
      the KSP object inside a SNES object. By indenting each lower level further the hierarchy of objects
      is very clear.

.seealso:  PetscObjectIncrementTabLevel()
@*/
PetscErrorCode  PetscObjectSetTabLevel(PetscObject obj,PetscInt tab)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  obj->tablevel = tab;
  PetscFunctionReturn(0);
}

/*@
   PetscObjectIncrementTabLevel - Sets the number of tabs that ASCII output for that object use based on
         the tablevel of another object. This should be called immediately after the object is created.

   Not Collective

   Input Parameters:
+  obj - any PETSc object where we are changing the tab
.  oldobj - the object providing the tab
-  tab - the increment that is added to the old objects tab

   Level: developer

    Notes:
    this is used to manage the output from options that are imbedded in other objects. For example
      the KSP object inside a SNES object. By indenting each lower level further the hierarchy of objects
      is very clear.

.seealso:   PetscObjectSetTabLevel(),  PetscObjectGetTabLevel()

@*/
PetscErrorCode  PetscObjectIncrementTabLevel(PetscObject obj,PetscObject oldobj,PetscInt tab)
{

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (oldobj) obj->tablevel = oldobj->tablevel + tab;
  else obj->tablevel = tab;
  PetscFunctionReturn(0);
}
