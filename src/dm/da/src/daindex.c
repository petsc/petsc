#ifndef lint
static char vcid[] = "$Id: daindex.c,v 1.3 1996/04/09 23:14:21 bsmith Exp curfman $";
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
.  n - the number of local elements, including ghost nodes (or PETSC_NULL)
.  idx - the global indices

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual for details.

.keywords: distributed array, get, global, indices, local to global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocal(), DALocalToLocal() 
@*/
int DAGetGlobalIndices(DA da, int *n,int **idx)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (n) *n   = da->Nl;
  *idx = da->idx;
  return 0;
}

