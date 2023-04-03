/* Additional functions in the DMProduct API, which are not part of the general DM API. */
#include <petsc/private/dmproductimpl.h>

/*@C
  DMProductGetDM - Get sub-`DM` associated with a given slot of a `DMPRODUCT`

  Not Collective

  Input Parameters:
+ dm - the` DMPRODUCT`
- slot - which dimension slot, in the range 0 to dim-1

  Output Parameter:
. subdm - the sub-`DM`

  Level: advanced

.seealso: `DMPRODUCT`, `DMProductSetDM()`
@*/
PETSC_EXTERN PetscErrorCode DMProductGetDM(DM dm, PetscInt slot, DM *subdm)
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

/*@C
  DMProductSetDM - Set sub-`DM` associated with a given slot of `DMPRODUCT`

  Not Collective

  Input Parameters:
+ dm - the `DMPRODUCT`
. slot - which dimension slot, in the range 0 to dim-1
- subdm - the sub-`DM`

  Level: advanced

  Note:
  This function does not destroy the provided sub-`DM`. You may safely destroy it after calling this function.

.seealso: `DMPRODUCT`, `DMProductGetDM()`, `DMProductSetDimensionIndex()`
@*/
PETSC_EXTERN PetscErrorCode DMProductSetDM(DM dm, PetscInt slot, DM subdm)
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

/*@C
  DMProductSetDimensionIndex - Set the dimension index associated with a given slot/sub-`DM`

  Not Collective

  Input Parameters:
+ dm - the `DMPRODUCT`
. slot - which dimension slot, in the range 0 to dim-1
- idx - the dimension index of the sub-`DM`

  Level: advanced

.seealso: `DMPRODUCT`
@*/
PETSC_EXTERN PetscErrorCode DMProductSetDimensionIndex(DM dm, PetscInt slot, PetscInt idx)
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
