#ifndef lint
static char vcid[] = "$Id: da2.c,v 1.32 1996/01/27 04:56:41 bsmith Exp $";
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

.seealso: DAGetDistributedVector()
@*/
int   DAGetLocalVector(DA da,Vec* l)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *l = da->local;
  return 0;
}

