#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pname.c,v 1.8 1997/07/09 20:51:14 balay Exp bsmith $";
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

