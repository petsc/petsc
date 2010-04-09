#define PETSC_DLL
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petscsys.h"  /*I   "petscsys.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetCookie"
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
PetscErrorCode PETSC_DLLEXPORT PetscObjectGetCookie(PetscObject obj,PetscCookie *cookie)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  *cookie = obj->cookie;
  PetscFunctionReturn(0);
}

