/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

/*@
  DMDAGetGhostCorners - Returns the global (`i`,`j`,`k`) indices of the lower left
  corner and size of the local region, including ghost points.

  Not Collective

  Input Parameter:
. da - the `DMDA`

  Output Parameters:
+ x - the corner index for the first dimension
. y - the corner index for the second dimension (only used in 2D and 3D problems)
. z - the corner index for the third dimension (only used in 3D problems)
. m - the width in the first dimension
. n - the width in the second dimension (only used in 2D and 3D problems)
- p - the width in the third dimension (only used in 3D problems)

  Level: beginner

  Notes:
  Any of `y`, `z`, `n`, and `p` can be passed in as `NULL` if not needed.

  The corner information is independent of the number of degrees of
  freedom per node set with the `DMDACreateXX()` routine. Thus the `x`, `y`, and `z`
  can be thought of as the lower left coordinates of the patch of values on process on a logical grid and `m`, `n`, and `p` as the
  extent of the patch. Where
  grid point has (potentially) several degrees of freedom.

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGetCorners()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMDAGetOwnershipRanges()`, `DMStagGetGhostCorners()`, `DMSTAG`
@*/
PetscErrorCode DMDAGetGhostCorners(DM da, PeOp PetscInt *x, PeOp PetscInt *y, PeOp PetscInt *z, PeOp PetscInt *m, PeOp PetscInt *n, PeOp PetscInt *p)
{
  PetscInt w;
  DM_DA   *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  /* since the xs, xe ... have all been multiplied by the number of degrees
     of freedom per cell, w = dd->w, we divide that out before returning.*/
  w = dd->w;
  if (x) *x = dd->Xs / w + dd->xo;
  if (y) *y = dd->Ys + dd->yo;
  if (z) *z = dd->Zs + dd->zo;
  if (m) *m = (dd->Xe - dd->Xs) / w;
  if (n) *n = (dd->Ye - dd->Ys);
  if (p) *p = (dd->Ze - dd->Zs);
  PetscFunctionReturn(PETSC_SUCCESS);
}
