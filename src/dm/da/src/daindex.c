#ifndef lint
static char vcid[] = "$Id: daindex.c,v 1.4 1996/06/21 03:16:32 curfman Exp bsmith $";
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
          DAGlobalToLocal(), DALocalToLocal(),DAGetAO()  
@*/
int DAGetGlobalIndices(DA da, int *n,int **idx)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (n) *n   = da->Nl;
  *idx = da->idx;
  return 0;
}


/*@C
   DAGetAO - Gets the application ordering context for a distributed array.
     The AO in this case maps to the natural grid ordering.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ao - the application ordering context for natural ordering


.keywords: distributed array, get, global, indices, local to global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocal(), DALocalToLocal(), DAGetGlobalIndices()
@*/
int DAGetAO(DA da, AO *ao)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *ao = da->ao;
  return 0;
}
