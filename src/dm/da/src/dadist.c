#ifndef lint
static char vcid[] = "$Id: dadist.c,v 1.4 1996/04/17 04:09:39 curfman Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

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
  return 0;
}

int DAGetGlobalToGlobal1_Private(DA da,int **gtog1)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *gtog1 = da->gtog1;
  return 0;
}

