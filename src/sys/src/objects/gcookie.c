#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: gcookie.c,v 1.14 1998/04/27 19:48:45 curfman Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetCookie"
/*@C
   PetscObjectGetCookie - Gets the cookie for any PetscObject, 

   Not Collective
   
   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetCookie((PetscObject) mat,&cookie);

   Output Parameter:
.  cookie - the cookie

.keywords: object, get, cookie
@*/
int PetscObjectGetCookie(PetscObject obj,int *cookie)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  *cookie = obj->cookie;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectExists"
/*@
   PetscObjectExists - Determines whether a PETSc object has been destroyed.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetCookie((PetscObject) mat,&exists);

   Output Parameter:
.  exists - 0 if object does not exist; 1 if object does exist.

.keywords: object, exists
@*/
int PetscObjectExists(PetscObject obj,int *exists)
{
  PetscFunctionBegin;
  *exists = 0;
  if (!obj) PetscFunctionReturn(0);
  if (obj->cookie >= PETSC_COOKIE && obj->cookie <= LARGEST_PETSC_COOKIE) *exists = 1;
  PetscFunctionReturn(0);
}

