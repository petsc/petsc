/*$Id: pgname.c,v 1.20 2000/04/09 04:34:38 bsmith Exp bsmith $*/

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectGetName"
/*@C
   PetscObjectGetName - Gets a string name associated with a PETSc object.

   Not Collective

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetName((PetscObject)mat,&name);
-  name - the name associated with obj

   Level: intermediate

.keywords: object, get, name

.seealso: PetscObjectSetName()
@*/
int PetscObjectGetName(PetscObject obj,char *name[])
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  if (!name) SETERRQ(PETSC_ERR_ARG_BADPTR,0,"Void location for name");
  *name = obj->name;
  PetscFunctionReturn(0);
}

