#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pname.c,v 1.9 1997/08/22 15:11:48 bsmith Exp bsmith $";
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
  PetscFunctionBegin;
  if (!obj) SETERRQ(1,0,"Null object");
  obj->name = name;
  PetscFunctionReturn(0);
}

