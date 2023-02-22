#include <petsc/private/dmproductimpl.h>

static PetscErrorCode DMDestroy_Product(DM dm)
{
  DM_Product *product = (DM_Product *)dm->data;
  PetscInt    d;

  PetscFunctionBeginUser;
  for (d = 0; d < DMPRODUCT_MAX_DIM; ++d) PetscCall(DMDestroy(&product->dm[d]));
  PetscCall(PetscFree(product));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  DMPRODUCT = "product" - a DM representing a local Cartesian product of other DMs

  For each of dim dimensions, stores a sub-DM (need not be unique) and a dimension index. This specifies
  which dimension of the sub-DM corresponds to each dimension of the DMProduct.

  Level: advanced

.seealso: `DM`, `DMSTAG`, `DMProductGetDM()`, `DMProductSetDimensionIndex()`, `DMProductSetDM()`, `DMStagSetUniformCoordinatesProduct()`,
          `DMStagGetProductCoordinateArrays()`, `DMStagGetProductCoordinateArraysRead()`
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Product(DM dm)
{
  DM_Product *product;
  PetscInt    d;

  PetscFunctionBegin;
  PetscValidPointer(dm, 1);
  PetscCall(PetscNew(&product));
  dm->data = product;

  for (d = 0; d < DMPRODUCT_MAX_DIM; ++d) product->dm[d] = NULL;
  for (d = 0; d < DMPRODUCT_MAX_DIM; ++d) product->dim[d] = -1;

  dm->ops->destroy = DMDestroy_Product;
  PetscFunctionReturn(PETSC_SUCCESS);
}
