#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dagtol.c,v 1.13 1998/05/29 20:39:21 bsmith Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DAGlobalToLocalBegin"
/*@
   DAGlobalToLocalBegin - Maps values from the global vector to the local
   patch; the ghost points are included. Must be followed by 
   DAGlobalToLocalEnd() to complete the exchange.

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local values

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, begin

.seealso: DAGlobalToLocalEnd(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalBegin(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  ierr = VecScatterBegin(g,l,mode,SCATTER_FORWARD,da->gtol); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DAGlobalToLocalEnd"
/*@
   DAGlobalToLocalEnd - Maps values from the global vector to the local
   patch; the ghost points are included. Must be preceeded by 
   DAGlobalToLocalBegin().

   Collective on DA

   Input Parameters:
+  da - the distributed array context
.  g - the global vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local values

   Notes:
   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, global to local, end

.seealso: DAGlobalToLocalBegin(), DALocalToGlobal(), DACreate2d()
@*/
int DAGlobalToLocalEnd(DA da,Vec g, InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  PetscValidHeaderSpecific(l,VEC_COOKIE);
  PetscValidHeaderSpecific(g,VEC_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_FORWARD,da->gtol); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

