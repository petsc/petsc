#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dadist.c,v 1.10 1997/08/22 15:18:43 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAGetDistributedVector"
/*@C
   DAGetDistributedVector - Gets a distributed vector for a 
   distributed array.  Additional vectors of the same type can be 
   created with VecDuplicate().

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed vector

.keywords: distributed array, get, global, distributed, vector

.seealso: DAGetLocalVector(), VecDuplicate(), VecDuplicateVecs()
@*/
int   DAGetDistributedVector(DA da,Vec* g)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *g = da->global;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGetGlobalToGlobal1_Private"
int DAGetGlobalToGlobal1_Private(DA da,int **gtog1)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *gtog1 = da->gtog1;
  PetscFunctionReturn(0);
}

