#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pgname.c,v 1.9 1997/07/09 20:51:14 balay Exp bsmith $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetName"
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

