#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dalocal.c,v 1.10 1997/10/19 03:30:13 bsmith Exp bsmith $";
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
  int ierr, cnt;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  /*
     If base vector is already in use then we duplicate it
  */
  ierr = PetscObjectGetReference((PetscObject)da->local,&cnt); CHKERRQ(ierr);
  if (cnt > 1) {
    ierr = VecDuplicate(da->local,g);CHKERRQ(ierr);
  } else {
    *g = da->local;
    ierr = PetscObjectReference((PetscObject)*g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
