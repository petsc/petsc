
#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/


/*@C
   DMDACreatePF - Creates an appropriately dimensioned PF mathematical function object
      from a DMDA.

   Collective on da

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  pf - the mathematical function object

   Level: advanced

   Not supported from Fortran

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMCreateGlobalVector()
@*/
PetscErrorCode  DMDACreatePF(DM da,PF *pf)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidPointer(pf,2);
  ierr = PFCreate(PetscObjectComm((PetscObject)da),da->dim,dd->w,pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


