
#ifndef lint
static char vcid[] = "$Id: pname.c,v 1.5 1997/01/01 03:36:26 bsmith Exp balay $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetName"
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
  if (!obj) SETERRQ(1,0,"Null object");
  obj->name = name;
  return 0;
}

