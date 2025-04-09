#include <petsc/private/dmproductimpl.h> /*I  "petsc/private/dmproductimpl.h"    I*/

/*@
  DMProductGetDM - Get sub-`DM` whose coordinates will be associated with a particular dimension of the `DMPRODUCT`

  Not Collective

  Input Parameters:
+ dm   - the` DMPRODUCT`
- slot - which dimension within `DMPRODUCT` whose coordinates is being provided, in the range 0 to $dim-1$

  Output Parameter:
. subdm - the sub-`DM`

  Level: advanced

  Note:
  You can call `DMProductGetDimensionIndex()` to determine which dimension in `subdm` is to be used to provide the coordinates, see `DMPRODUCT`

.seealso: `DMPRODUCT`, `DMProductSetDM()`, `DMProductGetDimensionIndex()`, `DMProductSetDimensionIndex()`
@*/
PetscErrorCode DMProductGetDM(DM dm, PetscInt slot, DM *subdm)
{
  DM_Product *product = (DM_Product *)dm->data;
  PetscInt    dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPRODUCT);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(slot < dim && slot >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "slot number must be in range 0-%" PetscInt_FMT, dim - 1);
  *subdm = product->dm[slot];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMProductSetDM - Set sub-`DM` whose coordinates will be associated with a particular dimension of the `DMPRODUCT`

  Not Collective

  Input Parameters:
+ dm    - the `DMPRODUCT`
. slot  - which dimension within `DMPRODUCT` whose coordinates is being provided, in the range 0 to $dim-1$
- subdm - the sub-`DM`

  Level: advanced

  Notes:
  This function does not destroy the provided sub-`DM`. You may safely destroy it after calling this function.

  You can call `DMProductSetDimensionIndex()` to determine which dimension in `subdm` is to be used to provide the coordinates, see `DMPRODUCT`

.seealso: `DMPRODUCT`, `DMProductGetDM()`, `DMProductSetDimensionIndex()`, `DMProductGetDimensionIndex()`
@*/
PetscErrorCode DMProductSetDM(DM dm, PetscInt slot, DM subdm)
{
  DM_Product *product = (DM_Product *)dm->data;
  PetscInt    dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPRODUCT);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(slot < dim && slot >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "slot number must be in range 0-%" PetscInt_FMT, dim - 1);
  PetscCall(PetscObjectReference((PetscObject)subdm));
  PetscCall(DMDestroy(&product->dm[slot]));
  product->dm[slot] = subdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMProductSetDimensionIndex - Set which dimension `idx` of the sub-`DM` coordinates will be used associated with the `DMPRODUCT` dimension `slot`

  Not Collective

  Input Parameters:
+ dm   - the `DMPRODUCT`
. slot - which dimension, in the range 0 to $dim-1$ you are providing to the `dm`
- idx  - the dimension of the sub-`DM` to use

  Level: advanced

.seealso: `DMPRODUCT`, `DMProductGetDM()`, `DMProductGetDimensionIndex()`
@*/
PetscErrorCode DMProductSetDimensionIndex(DM dm, PetscInt slot, PetscInt idx)
{
  DM_Product *product = (DM_Product *)dm->data;
  PetscInt    dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPRODUCT);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(slot < dim && slot >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "slot number must be in range 0-%" PetscInt_FMT, dim - 1);
  product->dim[slot] = idx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMProductGetDimensionIndex - Get which dimension `idx` of the sub-`DM` coordinates will be used associated with the `DMPRODUCT` dimension `slot`

  Not Collective

  Input Parameters:
+ dm   - the `DMPRODUCT`
- slot - which dimension, in the range 0 to $dim-1$ of `dm`

  Output Parameter:
. idx - the dimension of the sub-`DM`

  Level: advanced

.seealso: `DMPRODUCT`, `DMProductGetDM()`, `DMProductSetDimensionIndex()`
@*/
PetscErrorCode DMProductGetDimensionIndex(DM dm, PetscInt slot, PetscInt *idx)
{
  DM_Product *product = (DM_Product *)dm->data;
  PetscInt    dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPRODUCT);
  PetscCall(DMGetDimension(dm, &dim));
  PetscAssertPointer(idx, 3);
  PetscCheck(slot < dim && slot >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "slot number must be in range 0-%" PetscInt_FMT, dim - 1);
  *idx = product->dim[slot];
  PetscFunctionReturn(PETSC_SUCCESS);
}
