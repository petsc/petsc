

#ifndef lint
static char vcid[] = "$Id: pgname.c,v 1.7 1997/02/04 21:24:05 bsmith Exp bsmith $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetName" /* ADIC Ignore */
/*@C
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name associated with obj

.keywords: object, get, name

.seealso: PetscObjectSetName()
@*/
int PetscObjectGetName(PetscObject obj,char **name)
{
  if (!obj) SETERRQ(1,0,"Null object");
  if (!name) SETERRQ(1,0,"Void location for name");
  *name = obj->name;
  return 0;
}

