#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dalocal.c,v 1.13 1998/04/13 17:58:52 bsmith Exp curfman $";
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
   This is a regular PETSc vector that should be destroyed with 
   a call to VecDestroy().

.keywords: distributed array, get, local, distributed, vector

.seealso: DACreateGlobalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DACreateLocalVector(DA da,Vec* g)
{
  int ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecDuplicate(da->local,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
