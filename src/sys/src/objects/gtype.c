#ifndef lint
static char vcid[] = "$Id: gtype.c,v 1.2 1996/02/08 18:26:06 bsmith Exp curfman $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

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
  if (!obj) SETERRQ(1,"PetscObjectGetComm:Null object");
  *type = obj->type;
  return 0;
}



