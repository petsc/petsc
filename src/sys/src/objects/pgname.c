
#ifndef lint
static char vcid[] = "$Id: pgname.c,v 1.1 1996/01/30 19:09:30 bsmith Exp bsmith $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

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
  if (!obj) SETERRQ(1,"PetscObjectGetName:Null object");
  if (!name) SETERRQ(1,"PetscObjectGetName:Void location for name");
  *name = obj->name;
  return 0;
}

