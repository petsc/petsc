#ifndef lint
static char vcid[] = "$Id: daltol.c,v 1.3 1996/07/02 23:59:34 curfman Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

/*@
   DALocalToLocalBegin - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly. Must be followed by DALocalToLocalEnd().

   Input Parameters:
.  da - the distributed array context
.  g - the original local vector
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

.keywords: distributed array, local to local, begin

.seealso: DALocalToLocalEnd(), DALocalToGlobal(), DAGlobalToLocal()
@*/
int DALocalToLocalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecScatterBegin(g,l,mode,SCATTER_ALL,da->ltol); CHKERRQ(ierr);
  return 0;
}

/*@
   DALocalToLocalEnd - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly.  Must be preceeded by 
   DALocalToLocalBegin().

   Input Parameters:
.  da - the distributed array context
.  g - the original local vector
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

.keywords: distributed array, local to local, end

.seealso: DALocalToLocalBegin(), DALocalToGlobal(), DAGlobalToLocal()
@*/
int DALocalToLocalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_ALL,da->ltol); CHKERRQ(ierr);
  return 0;
}

