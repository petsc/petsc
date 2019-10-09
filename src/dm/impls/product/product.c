#include <petsc/private/dmproductimpl.h>

static PetscErrorCode DMDestroy_Product(DM dm)
{
  PetscErrorCode ierr;
  DM_Product     *product = (DM_Product*)dm->data;
  PetscInt       d;

  PetscFunctionBeginUser;
  for (d=0; d<DMPRODUCT_MAX_DIM; ++d) {
    ierr = DMDestroy(&product->dm[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(product);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  DMPRODUCT = "product" - a DM representing a local Cartesian product of other DMs

  For each of dim dimensions, stores a sub-DM (need not be unique) and a dimension index. This specifies
  which dimension of the sub-DM corresponds to each dimension of the DMProduct.

  Level: advanced

.seealso: DM, DMSTAG, DMProductGetDM(), DMProductSetDimensionIndex(), DMProductSetDM(), DMStagSetUniformCoordinatesProduct(),
          DMStagGet1dCoordinateArraysDOFRead()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Product(DM dm)
{
  PetscErrorCode ierr;
  DM_Product     *product;
  PetscInt       d;

  PetscFunctionBegin;
  PetscValidPointer(dm,1);
  ierr = PetscNewLog(dm,&product);CHKERRQ(ierr);
  dm->data = product;

  for (d=0; d<DMPRODUCT_MAX_DIM; ++d) product->dm[d]  = NULL;
  for (d=0; d<DMPRODUCT_MAX_DIM; ++d) product->dim[d] = -1;

  dm->ops->destroy            = DMDestroy_Product;
  PetscFunctionReturn(0);
}
