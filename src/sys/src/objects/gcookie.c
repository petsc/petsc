/*$Id: gcookie.c,v 1.21 2000/04/12 04:21:29 bsmith Exp bsmith $*/
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectGetCookie"
/*@C
   PetscObjectGetCookie - Gets the cookie for any PetscObject, 

   Not Collective
   
   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetCookie((PetscObject)mat,&cookie);

   Output Parameter:
.  cookie - the cookie

   Level: developer

@*/
int PetscObjectGetCookie(PetscObject obj,int *cookie)
{
  PetscFunctionBegin;
  if (!obj) SETERRQ(PETSC_ERR_ARG_CORRUPT,0,"Null object");
  *cookie = obj->cookie;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscObjectExists"
/*@
   PetscObjectExists - Determines whether a PETSc object has been destroyed.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetCookie((PetscObject)mat,&exists);

   Output Parameter:
.  exists - PETSC_FALSE if object does not exist; PETSC_TRUE if object does exist.

   Level: developer

@*/
int PetscObjectExists(PetscObject obj,PetscTruth *exists)
{
  PetscFunctionBegin;
  *exists = PETSC_FALSE;
  if (!obj) PetscFunctionReturn(0);
  if (obj->cookie >= PETSC_COOKIE && obj->cookie <= LARGEST_PETSC_COOKIE) *exists = PETSC_TRUE;
  PetscFunctionReturn(0);
}

