#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dalocal.c,v 1.16 1998/11/20 23:19:22 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

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

    *g = da->local;
    da->localused = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}
