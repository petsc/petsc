 
#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/


#undef __FUNCT__  
#define __FUNCT__ "DMDACreatePF"
/*@C
   DMDACreatePF - Creates an appropriately dimensioned PF mathematical function object
      from a DMDA.

   Collective on DMDA

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  pf - the mathematical function object

   Level: advanced


.keywords:  distributed array, grid function

.seealso: DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMDestroy(), DMCreateGlobalVector()
@*/
PetscErrorCode  DMDACreatePF(DM da,PF *pf)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(pf,2);
  ierr = PFCreate(((PetscObject)da)->comm,dd->dim,dd->w,pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 

