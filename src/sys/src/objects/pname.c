#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pname.c,v 1.13 1998/04/13 17:30:26 bsmith Exp curfman $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetName"
/*@C 
   PetscObjectSetName - Sets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject) mat,name);
-  name - the name to give obj

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

