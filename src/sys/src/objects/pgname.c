
#ifndef lint
static char vcid[] = "$Id: pgname.c,v 1.3 1996/12/16 21:56:31 balay Exp balay $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNCTION__  
#define __FUNCTION__ "PetscObjectGetName"
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
  if (!obj) SETERRQ(1,"Null object");
  if (!name) SETERRQ(1,"Void location for name");
  *name = obj->name;
  return 0;
}

