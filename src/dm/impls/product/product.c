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

static PetscErrorCode DMView_Product(DM dm, PetscViewer viewer)
{
  DM_Product *product = (DM_Product *)dm->data;
  PetscInt    d;

  PetscFunctionBegin;
  for (d = 0; d < DMPRODUCT_MAX_DIM; ++d) {
    if (product->dm[d]) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  DM that defines dimension %" PetscInt_FMT "\n", d));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(DMView(product->dm[d], viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  DMPRODUCT = "product" - a `DM` representing a local Cartesian product of other `DM`

  Level: advanced

  Notes:
  The `DM` is usually used for managing coordinates of other `DM` via `DMGetCoordinateDM()` and `DMSetCoordinateDM()`

  For each of `dim` dimensions, the `DMPRODUCT` contains a `DM` and a dimension index. The dimensional index, set with `DMProductSetDimensionIndex()`
  specifies  which dimension of the sub-`DM` coordinates corresponds to a particular dimension of the `DMPRODUCT`. For example,
.vb
  DM da1, da2;
  DM dm
  DMCreate(PETSC_COMM_WORLD,&dm);
  DMSetType(dm,DMPRODUCT);
  DMSetDimension(dm,3);
  DMProductSetDM(dm,0,da1);
  DMProductSetDimensionIndex(dm,0,0);
  DMProductSetDM(dm,1,da2);
  DMProductSetDimensionIndex(dm,1,0);
  DMProductSetDM(dm,2,da1);
  DMProductSetDimensionIndex(dm,2,1);
.ve
  results in a three-dimensional `DM` whose `x` coordinate values are obtained from the `x` coordinate values of `da1`, whose `y` coordinate values are obtained from
  the 'x' coordinate values of `da2` and whose `z` coordinate values are obtained from the `y` coordinate values of `da1`.

.seealso: `DM`, `DMSTAG`, `DMProductGetDM()`, `DMProductSetDimensionIndex()`, `DMProductSetDM()`, `DMStagSetUniformCoordinatesProduct()`,
          `DMStagGetProductCoordinateArrays()`, `DMStagGetProductCoordinateArraysRead()`, `DMGetCoordinateDM()`, `DMSetCoordinateDM()`
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Product(DM dm)
{
  DM_Product *product;
  PetscInt    d;

  PetscFunctionBegin;
  PetscAssertPointer(dm, 1);
  PetscCall(PetscNew(&product));
  dm->data = product;

  for (d = 0; d < DMPRODUCT_MAX_DIM; ++d) product->dm[d] = NULL;
  for (d = 0; d < DMPRODUCT_MAX_DIM; ++d) product->dim[d] = -1;

  dm->ops->destroy = DMDestroy_Product;
  dm->ops->view    = DMView_Product;
  PetscFunctionReturn(PETSC_SUCCESS);
}
