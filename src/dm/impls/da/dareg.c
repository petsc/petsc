#include <petsc/private/dmdaimpl.h>    /*I "petscdmda.h"  I*/

extern PetscErrorCode  DMSetUp_DA_1D(DM);
extern PetscErrorCode  DMSetUp_DA_2D(DM);
extern PetscErrorCode  DMSetUp_DA_3D(DM);

PetscErrorCode  DMSetUp_DA(DM da)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID,1);
  PetscAssertFalse(dd->w < 1,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %D",dd->w);
  PetscAssertFalse(dd->s < 0,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %D",dd->s);

  ierr = PetscCalloc1(dd->w+1,&dd->fieldname);CHKERRQ(ierr);
  ierr = PetscCalloc1(da->dim,&dd->coordinatename);CHKERRQ(ierr);
  if (da->dim == 1) {
    ierr = DMSetUp_DA_1D(da);CHKERRQ(ierr);
  } else if (da->dim == 2) {
    ierr = DMSetUp_DA_2D(da);CHKERRQ(ierr);
  } else if (da->dim == 3) {
    ierr = DMSetUp_DA_3D(da);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"DMs only supported for 1, 2, and 3d");
  ierr = DMViewFromOptions(da,NULL,"-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
