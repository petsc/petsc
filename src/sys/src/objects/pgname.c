#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pgname.c,v 1.14 1998/04/13 17:30:26 bsmith Exp curfman $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetName"
/*@C
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetName((PetscObject) mat,&name);
-  name - the name associated with obj

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

