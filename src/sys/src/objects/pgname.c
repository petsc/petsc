#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pgname.c,v 1.13 1998/03/20 22:47:23 bsmith Exp bsmith $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetName"
/*@C
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Input Parameters:
.  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetName((PetscObject) mat,&name);
.  name - the name associated with obj

   Not Collective

.keywords: object, get, name

.seealso: PetscObjectSetName()
@*/
int PetscObjectGetName(PetscObject obj,char **name)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  if (!name) SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Void location for name");
  *name = obj->name;
  PetscFunctionReturn(0);
}

