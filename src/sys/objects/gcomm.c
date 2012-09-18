
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petscsys.h>  /*I   "petscsys.h"    I*/

#undef __FUNCT__
#define __FUNCT__ "PetscObjectGetComm"
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

   Concepts: communicator^getting from object
   Concepts: MPI communicator^getting from object

@*/
PetscErrorCode  PetscObjectGetComm(PetscObject obj,MPI_Comm *comm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidPointer(comm,2);
  if (obj->bops->getcomm) {
    ierr = obj->bops->getcomm(obj,comm);CHKERRQ(ierr);
  } else {
    *comm = obj->comm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectGetTabLevel"
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

    Notes: this is used to manage the output from options that are imbedded in other objects. For example
      the KSP object inside a SNES object. By indenting each lower level further the heirarchy of objects
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

#undef __FUNCT__
#define __FUNCT__ "PetscObjectSetTabLevel"
/*@
   PetscObjectSetTabLevel - Sets the number of tabs that ASCII output for that object use

   Not Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example,
         PetscObjectGetComm((PetscObject)mat,&comm);
-   tab - the number of tabs

   Level: developer

    Notes: this is used to manage the output from options that are imbedded in other objects. For example
      the KSP object inside a SNES object. By indenting each lower level further the heirarchy of objects
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

#undef __FUNCT__
#define __FUNCT__ "PetscObjectIncrementTabLevel"
/*@
   PetscObjectIncrementTabLevel - Sets the number of tabs that ASCII output for that object use based on
         the tablevel of another object. This should be called immediately after the object is created.

   Not Collective

   Input Parameter:
+  obj - any PETSc object where we are changing the tab
.  oldobj - the object providing the tab
-  tab - the increment that is added to the old objects tab


   Level: developer

    Notes: this is used to manage the output from options that are imbedded in other objects. For example
      the KSP object inside a SNES object. By indenting each lower level further the heirarchy of objects
      is very clear.

.seealso:   PetscObjectSetLabLevel(),  PetscObjectGetTabLevel()

@*/
PetscErrorCode  PetscObjectIncrementTabLevel(PetscObject obj,PetscObject oldobj,PetscInt tab)
{

  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  if (oldobj) {
    obj->tablevel = oldobj->tablevel + tab;
  } else {
    obj->tablevel = tab;
  }
  PetscFunctionReturn(0);
}
