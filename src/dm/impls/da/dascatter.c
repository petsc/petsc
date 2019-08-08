
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/
extern PetscErrorCode DMLocalToLocalCreate_DA(DM);

/*@C
   DMDAGetScatter - Gets the global-to-local, and
   local-to-local vector scatter contexts for a distributed array.

   Collective on da

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  gtol - global-to-local scatter context (may be NULL)
-  ltol - local-to-local scatter context (may be NULL)

   Level: developer

   Notes:
   The output contexts are valid only as long as the input da is valid.
   If you delete the da, the scatter contexts will become invalid.

.seealso: DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()
@*/
PetscErrorCode  DMDAGetScatter(DM da,VecScatter *gtol,VecScatter *ltol)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  if (gtol) *gtol = dd->gtol;
  if (ltol) {
    if (!dd->ltol) {
      ierr = DMLocalToLocalCreate_DA(da);CHKERRQ(ierr);
    }
    *ltol = dd->ltol;
  }
  PetscFunctionReturn(0);
}

