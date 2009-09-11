#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/
EXTERN PetscErrorCode DALocalToLocalCreate(DA);

#undef __FUNCT__  
#define __FUNCT__ "DAGetScatter"
/*@C
   DAGetScatter - Gets the local-to-global, local-to-global, and 
   local-to-local vector scatter contexts for a distributed array.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  ltog - local-to-global scatter context (may be PETSC_NULL)
.  gtol - global-to-local scatter context (may be PETSC_NULL) 
-  ltol - local-to-local scatter context (may be PETSC_NULL)

   Level: developer

   Notes:
   The output contexts are valid only as long as the input da is valid.
   If you delete the da, the scatter contexts will become invalid.

.keywords: distributed array, get, scatter, context, global-to-local,
           local-to-global, local-to-local

.seealso: DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetScatter(DA da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (ltog) *ltog = da->ltog;
  if (gtol) *gtol = da->gtol;
  if (ltol) {
    if (!da->ltol) {
      ierr = DALocalToLocalCreate(da);CHKERRQ(ierr);
    }
    *ltol = da->ltol;
  }
  PetscFunctionReturn(0);
}
 
