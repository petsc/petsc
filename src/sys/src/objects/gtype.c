#define PETSC_DLL
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetType"
/*@C
   PetscObjectGetType - Gets the object type of any PetscObject.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetType((PetscObject)mat,&type);

   Output Parameter:
.  type - the object type

   Note: This is being PHASED out; all classes derived from abstract classes instead have a
         type_name

   Level: advanced

   Concepts: object type

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectGetType(PetscObject obj,int *type)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null object");
  *type = obj->type;
  PetscFunctionReturn(0);
}



