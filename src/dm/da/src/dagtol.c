#ifndef lint
static char vcid[] = "$Id: dagtol.c,v 1.3 1996/08/08 14:47:19 bsmith Exp curfman $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

/*@
   DAGlobalToLocalBegin - Maps values from the global vector to the local
   patch; the ghost points are included. Must be followed by 
   DAGlobalToLocalEnd() to complete the exchange.

   Input Parameters:
.  da - the distributed array context
.  g - the global vector
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local values

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToLocalEnd(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecScatterBegin(g,l,mode,SCATTER_ALL,da->gtol); CHKERRQ(ierr);
  return 0;
}

/*@
   DAGlobalToLocalEnd - Maps values from the global vector to the local
   patch; the ghost points are included. Must be preceeded by 
   DAGlobalToLocalBegin().

   Input Parameters:
.  da - the distributed array context
.  g - the global vector
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local values

.keywords: distributed array, global to local, end

.seealso: DAGlobalToLocalBegin(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_ALL,da->gtol); CHKERRQ(ierr);
  return 0;
}

