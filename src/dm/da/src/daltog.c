#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: daltog.c,v 1.8 1997/07/09 21:00:44 balay Exp bsmith $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/da/daimpl.h"    /*I   "da.h"   I*/

#undef __FUNC__  
#define __FUNC__ "DALocalToGlobal"
/*@
   DALocalToGlobal - Maps values from the local patch back to the 
   global vector. The ghost points are discarded.

   Input Parameters:
.  da - the distributed array context
.  l  - the local values
.  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  g - the global vector

.keywords: distributed array, local-to-global

.seealso: DAGlobalToLocalBegin(), DACreate2d()
@*/
int DALocalToGlobal(DA da,Vec l, InsertMode mode,Vec g)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecScatterBegin(l,g,mode,SCATTER_FORWARD,da->ltog); CHKERRQ(ierr);
  ierr = VecScatterEnd(l,g,mode,SCATTER_FORWARD,da->ltog); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








