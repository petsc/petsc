/*$Id: dalocal.c,v 1.20 1999/03/17 23:25:10 bsmith Exp bsmith $*/
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DACreateLocalVector"
/*@C
   DACreateLocalVector - Creates a parallel PETSc vector that
   may be used with the DAXXX routines.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the local vector

   Level: beginner

   Note:
   The output parameter, g, is a regular PETSc vector that should be destroyed
   with a call to VecDestroy() when usage is finished.

.keywords: distributed array, create, local, vector

.seealso: DACreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DACreateLocalVector(DA da,Vec* g)
{
  int ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (da->localused) {
    ierr = VecDuplicate(da->local,g);CHKERRQ(ierr);
  } else {
    /* 
     compose the DA into the vectors so they have access to the 
     distribution information. 
    */
    ierr = PetscObjectCompose((PetscObject)da->local,"DA",(PetscObject)da);CHKERRQ(ierr);
    da->localused = PETSC_TRUE;
    *g = da->local;
  }
  PetscFunctionReturn(0);
}
