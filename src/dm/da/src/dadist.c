
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dadist.c,v 1.17 1998/12/03 04:06:11 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/


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
   The output parameter, g, is a regular PETSc vector that should be destroyed
   with a call to VecDestroy() when usage is finished.

.keywords: distributed array, create, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DACreateGlobalVector(DA da,Vec* g)
{
  int ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (da->globalused) {
    ierr = VecDuplicate(da->global,g);CHKERRQ(ierr);
  } else {

    *g = da->global;
    da->globalused = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}



