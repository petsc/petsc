#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DALocalToGlobal"
/*@
   DALocalToGlobal - Maps values from the local patch back to the 
   global vector. The ghost points are discarded.

   Not Collective

   Input Parameters:
+  da - the distributed array context
.  l  - the local values
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  g - the global vector

   Level: beginner

   Note:
   This routine discards the values in the ghost point locations. Use 
   DALocalToGlobalBegin()/DALocalToGlobalEnd() to add the values from the
   ghost points.

   The global and local vectors used here need not be the same as those
   obtained from DACreateGlobalVector() and DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, local-to-global

.seealso: DAGlobalToLocalBegin(), DACreate2d(), DALocalToLocalBegin(),
           DALocalToLocalEnd(), DALocalToGlobalBegin(), DALocalToGlobalEnd()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DALocalToGlobal(DA da,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  ierr = VecScatterBegin(dd->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(dd->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








