#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gtype.c,v 1.15 1998/04/27 19:48:45 curfman Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetType"
/*@C
   PetscObjectGetType - Gets the object type of any PetscObject.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetType((PetscObject) mat,&type);

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



