#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dadist.c,v 1.14 1998/04/13 17:58:52 bsmith Exp curfman $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/


#undef __FUNC__  
#define __FUNC__ "DAGetGlobalToGlobal1_Private"
int DAGetGlobalToGlobal1_Private(DA da,int **gtog1)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *gtog1 = da->gtog1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DACreateGlobalVector"
/*@C
   DACreateGlobalVector - Creates a parallel PETSc vector that
   may be used with the DAXXX routines.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed global vector

   Note:
   This is a regular PETSc vector that should be destroyed with 
   a call to VecDestroy().

.keywords: distributed array, get, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DACreateGlobalVector(DA da,Vec* g)
{
  int ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecDuplicate(da->global,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

