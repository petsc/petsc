#ifndef lint
static char vcid[] = "$Id: gtype.c,v 1.7 1997/01/06 20:22:55 balay Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetType" /* ADIC Ignore */
/*@C
   PetscObjectGetType - Gets the object type of any PetscObject.

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameter:
.  type - the object type

.keywords: object, get, type
@*/
int PetscObjectGetType(PetscObject obj,int *type)
{
  if (!obj) SETERRQ(1,0,"Null object");
  *type = obj->type;
  return 0;
}



