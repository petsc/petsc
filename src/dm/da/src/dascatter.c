#ifndef lint
static char vcid[] = "$Id: dascatter.c,v 1.3 1996/04/20 04:21:49 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

/*@C
   DAGetScatter - Gets the local to global, local to global, and 
       local to local vector scatter contexts for a distributed array.

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ltog - local to global scatter context (may be PETSC_NULL)
.  gtol - global to local scatter context (may be PETSC_NULL) 
.  ltol - local to local scatter context (may be PETSC_NULL)

.keywords: distributed array, get, scatter, context, global to local,
           local to global

.seealso: DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
int DAGetScatter(DA da, VecScatter *ltog,VecScatter *gtol,VecScatter *ltol)
{
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (ltog) *ltog = da->ltog;
  if (gtol) *gtol = da->gtol;
  if (ltol) *ltol = da->ltol;
  return 0;
}
 
