#ifndef lint
static char vcid[] = "$Id: destroy.c,v 1.31 1997/01/06 20:22:55 balay Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectDestroy" /* ADIC Ignore */
/*@C
   PetscObjectDestroy - Destroys any PetscObject, regardless of the type. 
   This routine should seldom be needed.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

.keywords: object, destroy
@*/
int PetscObjectDestroy(PetscObject obj)
{
  if (!obj) SETERRQ(1,0,"Null object");
  if (obj->destroy) return (*obj->destroy)(obj);
  return 0;
}
