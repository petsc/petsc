
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include <petscsys.h>  /*I   "petscsys.h"    I*/

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

   Level: advanced

   Concepts: object type
@*/
PetscErrorCode  PetscObjectGetType(PetscObject obj, const char *type[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidPointer(type,2);
  *type = obj->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscObjectSetType"
/*@C
   PetscObjectSetType - Sets the object type of any PetscObject.

   Not Collective

   Input Parameters:
+  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example,
         PetscObjectGetType((PetscObject)mat,&type);
-  type - the object type

   Note: This does not currently work since we need to dispatch by type.

   Level: advanced

   Concepts: object type
@*/
PetscErrorCode  PetscObjectSetType(PetscObject obj, const char type[])
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  PetscValidCharPointer(type,2);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Cannot set the type of a generic PetscObject");
}
