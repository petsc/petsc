#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: daltog.c,v 1.10 1998/03/20 22:53:07 bsmith Exp bsmith $";
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

   Not Collective

   Note:
   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

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








