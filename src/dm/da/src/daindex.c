#ifndef lint
static char vcid[] = "$Id: da2.c,v 1.32 1996/01/27 04:56:41 bsmith Exp $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@C
   DAGetGlobalIndices - Returns the global node number of all local nodes,
   including ghost nodes.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  n - the number of local elements, including ghost nodes
.  idx - the global indices

.keywords: distributed array, get, global, indices, local to global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocal(), DALocalToLocal() 
@*/
int DAGetGlobalIndices(DA da, int *n,int **idx)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  *n   = da->Nl;
  *idx = da->idx;
  return 0;
}

