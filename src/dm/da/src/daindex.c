#ifndef lint
static char vcid[] = "$Id: daindex.c,v 1.11 1997/02/22 02:29:24 bsmith Exp curfman $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAGetGlobalIndices" /* ADIC Ignore */
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

.keywords: distributed array, get, global, indices, local-to-global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocal(), DALocalToLocal(), DAGetAO(), DAGetGlobalIndicesF90()
@*/
int DAGetGlobalIndices(DA da, int *n,int **idx)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (n) *n   = da->Nl;
  *idx = da->idx;
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "DAGetAO" /* ADIC Ignore */
/*@C
   DAGetAO - Gets the application ordering context for a distributed array.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ao - the application ordering context for DAs

   Notes:
   In this case, the AO maps to the natural grid ordering that would be used
   for the DA if only 1 processor were employed (ordering most rapidly in the
   x-direction, then y, then z).  Multiple degrees of freedom are numbered
   for each node (rather than 1 component for the whole grid, then the next
   component, etc.)

.keywords: distributed array, get, global, indices, local-to-global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocal(), DALocalToLocal(), DAGetGlobalIndices()
@*/
int DAGetAO(DA da, AO *ao)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  *ao = da->ao;
  return 0;
}

/*MC
    DAGetGlobalIndicesF90 - Returns a Fortran90 pointer to the list of 
    global indices (global node number of all local nodes, including
    ghost nodes).

    Input Parameter:
.   da - the distributed array

    Output Parameters:
.   n - the number of local elements, including ghost nodes (or PETSC_NULL)
.   idx - the Fortran90 pointer to the global indices
.   ierr - error code

    Synopsis:
    DAGetGlobalIndicesF90(DA da,integer n,{Scalar, pointer :: idx(:)},integer ierr)

    Notes:
    Currently only supported using the NAG F90 compiler.

.keywords: distributed array, get, global, indices, local-to-global, f90

.seealso: DAGetGlobalIndices()
M*/
