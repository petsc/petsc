#define PETSCDM_DLL
 
#include "private/daimpl.h"    /*I   "petscda.h"   I*/


#undef __FUNCT__  
#define __FUNCT__ "DACreatePF"
/*@C
   DACreatePF - Creates an appropriately dimensioned PF mathematical function object
      from a DA.

   Collective on DA

   Input Parameter:
.  da - initial distributed array

   Output Parameter:
.  pf - the mathematical function object

   Level: advanced


.keywords:  distributed array, grid function

.seealso: DACreate1d(), DACreate2d(), DACreate3d(), DMDestroy(), DMCreateGlobalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreatePF(DM da,PF *pf)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(pf,2);
  ierr = PFCreate(((PetscObject)da)->comm,dd->dim,dd->w,pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 

