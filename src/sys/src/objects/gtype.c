#ifndef lint
static char vcid[] = "$Id: gtype.c,v 1.1 1996/01/30 19:22:50 bsmith Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

/*@C
   PetscObjectGetType - Gets the type for any PetscObject, 

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameter:
.  type - the type

.keywords: object, get, type
@*/
int PetscObjectGetType(PetscObject obj,int *type)
{
  if (!obj) SETERRQ(1,"PetscObjectGetComm:Null object");
  *type = obj->type;
  return 0;
}



