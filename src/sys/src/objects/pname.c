
#ifndef lint
static char vcid[] = "$Id: pname.c,v 1.3 1996/12/16 21:56:41 balay Exp balay $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNCTION__  
#define __FUNCTION__ "PetscObjectSetName"
/*@C 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
.  name - the name to give obj

.keywords: object, set, name

.seealso: PetscObjectGetName()
@*/
int PetscObjectSetName(PetscObject obj,char *name)
{
  if (!obj) SETERRQ(1,"Null object");
  obj->name = name;
  return 0;
}

