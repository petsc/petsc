
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

/*
   DMLocalToLocalCreate_DA - Creates the local to local scatter

   Collective

   Input Parameter:
.  da - the distributed array

*/
PetscErrorCode DMLocalToLocalCreate_DA(DM da)
{
  PetscInt *idx, left, j, count, up, down, i, bottom, top, k, dim = da->dim;
  DM_DA    *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);

  if (dd->ltol) PetscFunctionReturn(PETSC_SUCCESS);
  /*
     We simply remap the values in the from part of
     global to local to read from an array with the ghost values
     rather then from the plain array.
  */
  PetscCall(VecScatterCopy(dd->gtol, &dd->ltol));
  if (dim == 1) {
    left = dd->xs - dd->Xs;
    PetscCall(PetscMalloc1(dd->xe - dd->xs, &idx));
    for (j = 0; j < dd->xe - dd->xs; j++) idx[j] = left + j;
  } else if (dim == 2) {
    left = dd->xs - dd->Xs;
    down = dd->ys - dd->Ys;
    up   = down + dd->ye - dd->ys;
    PetscCall(PetscMalloc1((dd->xe - dd->xs) * (up - down), &idx));
    count = 0;
    for (i = down; i < up; i++) {
      for (j = 0; j < dd->xe - dd->xs; j++) idx[count++] = left + i * (dd->Xe - dd->Xs) + j;
    }
  } else if (dim == 3) {
    left   = dd->xs - dd->Xs;
    bottom = dd->ys - dd->Ys;
    top    = bottom + dd->ye - dd->ys;
    down   = dd->zs - dd->Zs;
    up     = down + dd->ze - dd->zs;
    count  = (dd->xe - dd->xs) * (top - bottom) * (up - down);
    PetscCall(PetscMalloc1(count, &idx));
    count = 0;
    for (i = down; i < up; i++) {
      for (j = bottom; j < top; j++) {
        for (k = 0; k < dd->xe - dd->xs; k++) idx[count++] = (left + j * (dd->Xe - dd->Xs)) + i * (dd->Xe - dd->Xs) * (dd->Ye - dd->Ys) + k;
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_CORRUPT, "DMDA has invalid dimension %" PetscInt_FMT, dim);

  PetscCall(VecScatterRemap(dd->ltol, idx, NULL));
  PetscCall(PetscFree(idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMLocalToLocalBegin_DA(DM da, Vec g, InsertMode mode, Vec l)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  if (!dd->ltol) PetscCall(DMLocalToLocalCreate_DA(da));
  PetscCall(VecScatterBegin(dd->ltol, g, l, mode, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMLocalToLocalEnd_DA(DM da, Vec g, InsertMode mode, Vec l)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 2);
  PetscCall(VecScatterEnd(dd->ltol, g, l, mode, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}
