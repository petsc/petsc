/*$Id: gtype.c,v 1.19 2000/04/09 04:34:38 bsmith Exp bsmith $*/
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectGetType"
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

.keywords: object, get, type
@*/
int PetscObjectGetType(PetscObject obj,int *type)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  *type = obj->type;
  PetscFunctionReturn(0);
}



