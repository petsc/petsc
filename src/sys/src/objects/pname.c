#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pname.c,v 1.15 1998/07/23 22:46:55 bsmith Exp balay $";
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
int PetscObjectSetName(PetscObject obj,const char name[])
{
  int len;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  len = PetscStrlen(name);
  obj->name = (char *)PetscMalloc(sizeof(char)*(len+1)); CHKPTRQ(obj->name);
  PetscStrcpy(obj->name,name);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectPublish"
/*@C 
   PetscObjectPublish - Publishs an object for the ALICE Memory Snooper

   Collective on PetscObject

   Input Parameters:
+  obj - the Petsc variable
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectSetName((PetscObject) mat,name);

.keywords: object, monitoring, publishing

.seealso: PetscObjectSetName()
@*/
int PetscObjectPublish(PetscObject obj)
{
  int ierr;

  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  if (obj->bops->publish) {
    ierr = (*obj->bops->publish)(obj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
