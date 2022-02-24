#include <petsc/private/dmdaimpl.h>    /*I "petscdmda.h"  I*/

extern PetscErrorCode  DMSetUp_DA_1D(DM);
extern PetscErrorCode  DMSetUp_DA_2D(DM);
extern PetscErrorCode  DMSetUp_DA_3D(DM);

PetscErrorCode  DMSetUp_DA(DM da)
{
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID,1);
  PetscCheckFalse(dd->w < 1,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Must have 1 or more degrees of freedom per node: %D",dd->w);
  PetscCheckFalse(dd->s < 0,PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_OUTOFRANGE,"Stencil width cannot be negative: %D",dd->s);

  CHKERRQ(PetscCalloc1(dd->w+1,&dd->fieldname));
  CHKERRQ(PetscCalloc1(da->dim,&dd->coordinatename));
  if (da->dim == 1) {
    CHKERRQ(DMSetUp_DA_1D(da));
  } else if (da->dim == 2) {
    CHKERRQ(DMSetUp_DA_2D(da));
  } else if (da->dim == 3) {
    CHKERRQ(DMSetUp_DA_3D(da));
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_SUP,"DMs only supported for 1, 2, and 3d");
  CHKERRQ(DMViewFromOptions(da,NULL,"-dm_view"));
  PetscFunctionReturn(0);
}
