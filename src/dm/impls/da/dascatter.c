
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/
extern PetscErrorCode DMDALocalToLocalCreate(DM);

#undef __FUNCT__
#define __FUNCT__ "DMDAGetScatter"
/*@C
   DMDAGetScatter - Gets the local-to-global, local-to-global, and
   local-to-local vector scatter contexts for a distributed array.

   Collective on DMDA

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

.seealso: DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMLocalToGlobalBegin()
@*/
PetscErrorCode  DMDAGetScatter(DM da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (ltog) *ltog = dd->ltog;
  if (gtol) *gtol = dd->gtol;
  if (ltol) {
    if (!dd->ltol) {
      ierr = DMDALocalToLocalCreate(da);CHKERRQ(ierr);
    }
    *ltol = dd->ltol;
  }
  PetscFunctionReturn(0);
}

