#include <petsc/private/dmdaimpl.h> /*I "petscdmda.h"  I*/

extern PetscErrorCode DMSetUp_DA_1D(DM);
extern PetscErrorCode DMSetUp_DA_2D(DM);
extern PetscErrorCode DMSetUp_DA_3D(DM);

PetscErrorCode DMSetUp_DA(DM da)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscCheck(dd->w >= 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Must have 1 or more degrees of freedom per node: %" PetscInt_FMT, dd->w);
  PetscCheck(dd->s >= 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Stencil width cannot be negative: %" PetscInt_FMT, dd->s);

  PetscCall(PetscCalloc1(dd->w + 1, &dd->fieldname));
  PetscCall(PetscCalloc1(da->dim, &dd->coordinatename));
  if (da->dim == 1) {
    PetscCall(DMSetUp_DA_1D(da));
  } else if (da->dim == 2) {
    PetscCall(DMSetUp_DA_2D(da));
  } else if (da->dim == 3) {
    PetscCall(DMSetUp_DA_3D(da));
  } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "DMs only supported for 1, 2, and 3d");
  PetscCall(DMViewFromOptions(da, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}
