#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pname.c,v 1.12 1998/03/20 22:47:23 bsmith Exp bsmith $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetName"
/*@C 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject) mat,name);
.  name - the name to give obj

   Not Collective

.keywords: object, set, name

.seealso: PetscObjectGetName()
@*/
int PetscObjectSetName(PetscObject obj,char *name)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  obj->name = name;
  PetscFunctionReturn(0);
}

