#ifndef lint
static char vcid[] = "$Id: da2.c,v 1.32 1996/01/27 04:56:41 bsmith Exp $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@C
   DAGetDistributedVector - Gets a distributed vector for a 
   distributed array.  Additional vectors of the same type can be 
   created with VecDuplicate().

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed vector

.keywords: distributed array, get, global, distributed, vector

.seealso: DAGetLocalVector()
@*/
int   DAGetDistributedVector(DA da,Vec* g)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *g = da->global;
  return 0;
}

