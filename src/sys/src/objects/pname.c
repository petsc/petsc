#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pname.c,v 1.7 1997/02/22 02:23:29 bsmith Exp balay $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetName" /* ADIC Ignore */
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

