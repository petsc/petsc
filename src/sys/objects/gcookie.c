#define PETSC_DLL
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petscsys.h"  /*I   "petscsys.h"    I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectGetClassid"
/*@C
   PetscObjectGetClassid - Gets the classid for any PetscObject, 

   Not Collective
   
   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetClassid((PetscObject)mat,&classid);

   Output Parameter:
.  classid - the classid

   Level: developer

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectGetClassid(PetscObject obj,PetscClassId *classid)
{
  PetscFunctionBegin;
  PetscValidHeader(obj,1);
  *classid = obj->classid;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectExists"
/*@
   PetscObjectExists - Determines whether a PETSc object has been destroyed.

   Not Collective

   Input Parameter:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
         Thus must be cast with a (PetscObject), for example, 
         PetscObjectGetClassid((PetscObject)mat,&exists);

   Output Parameter:
.  exists - PETSC_FALSE if object does not exist; PETSC_TRUE if object does exist.

   Level: developer

@*/
PetscErrorCode PETSC_DLLEXPORT PetscObjectExists(PetscObject obj,PetscTruth *exists)
{
  PetscFunctionBegin;
  *exists = PETSC_FALSE;
  if (!obj) PetscFunctionReturn(0);
  if (obj->classid >= PETSC_SMALLEST_CLASSID && obj->classid <= PETSC_LARGEST_CLASSID) *exists = PETSC_TRUE;
  PetscFunctionReturn(0);
}

