/*$Id: gcomm.c,v 1.25 2001/03/23 23:20:38 balay Exp $*/
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetState"
/*@C
   PetscObjectGetState - Gets the state of any PetscObject, 
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example, 
         PetscObjectGetState((PetscObject)mat,&state);

   Output Parameter:
.  state - the object state

   Notes: object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is only maintained for Mat objects.

   Level: advanced

   seealso: PetscObjectIncreaseState

   Concepts: state

@*/
int PetscObjectGetState(PetscObject obj,int *state)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  *state = obj->state;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectIncreaseState"
/*@C
   PetscObjectIncreaseState - Increases the state of any PetscObject, 
   regardless of the type.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP. Thus must be
         cast with a (PetscObject), for example, 
         PetscObjectIncreaseState((PetscObject)mat);

   Notes: object state is an integer which gets increased every time
   the object is changed. By saving and later querying the object state
   one can determine whether information about the object is still current.
   Currently, state is only maintained for Mat objects.

   This routine is mostly for internal use by PETSc; a developer need only
   call it after explicit access to an object's internals. Routines such
   as VecSet or MatScale already call this routine.

   Level: developer

   seealso: PetscObjectGetState

   Concepts: state

@*/
int PetscObjectIncreaseState(PetscObject obj)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  obj->state++;
  PetscFunctionReturn(0);
}
