#ifndef lint
static char vcid[] = "$Id: dalocal.c,v 1.5 1996/12/16 21:10:30 balay Exp balay $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAGetLocalVector"
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

