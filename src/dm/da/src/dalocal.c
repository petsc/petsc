#ifndef lint
static char vcid[] = "$Id: dalocal.c,v 1.2 1996/02/22 00:31:35 curfman Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@C
   DAGetLocalVector - Gets a local vector (including ghost points) for a 
   distributed array.  Additional vectors of the same type can be created 
   with VecDuplicate().

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  l - the distributed vector

.keywords: distributed array, get, local, vector

.seealso: DAGetDistributedVector(), VecDuplicate(), VecDuplicateVecs()
@*/
int   DAGetLocalVector(DA da,Vec* l)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *l = da->local;
  return 0;
}

