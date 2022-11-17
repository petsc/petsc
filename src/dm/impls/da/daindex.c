/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

/*
   Gets the natural number for each global number on the process.

   Used by DMDAGetAO() and DMDAGlobalToNatural_Create()
*/
PetscErrorCode DMDAGetNatural_Private(DM da, PetscInt *outNlocal, IS *isnatural)
{
  PetscInt Nlocal, i, j, k, *lidx, lict = 0, dim = da->dim;
  DM_DA   *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  Nlocal = (dd->xe - dd->xs);
  if (dim > 1) Nlocal *= (dd->ye - dd->ys);
  if (dim > 2) Nlocal *= (dd->ze - dd->zs);

  PetscCall(PetscMalloc1(Nlocal, &lidx));

  if (dim == 1) {
    for (i = dd->xs; i < dd->xe; i++) {
      /*  global number in natural ordering */
      lidx[lict++] = i;
    }
  } else if (dim == 2) {
    for (j = dd->ys; j < dd->ye; j++) {
      for (i = dd->xs; i < dd->xe; i++) {
        /*  global number in natural ordering */
        lidx[lict++] = i + j * dd->M * dd->w;
      }
    }
  } else if (dim == 3) {
    for (k = dd->zs; k < dd->ze; k++) {
      for (j = dd->ys; j < dd->ye; j++) {
        for (i = dd->xs; i < dd->xe; i++) lidx[lict++] = i + j * dd->M * dd->w + k * dd->M * dd->N * dd->w;
      }
    }
  }
  *outNlocal = Nlocal;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)da), Nlocal, lidx, PETSC_OWN_POINTER, isnatural));
  PetscFunctionReturn(0);
}

/*@C
   DMDASetAOType - Sets the type of application ordering for a distributed array.

   Collective on da

   Input Parameters:
+  da - the distributed array
-  aotype - type of `AO`

   Output Parameters:

   Level: intermediate

   Note:
   It will generate and error if an `AO` has already been obtained with a call to `DMDAGetAO()` and the user sets a different `AOType`

.seealso: `DM`, `DMDA`, `DMDACreate2d()`, `DMDAGetAO()`, `DMDAGetGhostCorners()`, `DMDAGetCorners()`, `DMLocalToGlobal()`
          `DMGlobalToLocalBegin()`, `DMGlobalToLocalEnd()`, `DMLocalToLocalBegin()`, `DMLocalToLocalEnd()`, `DMDAGetGlobalIndices()`, `DMDAGetOwnershipRanges()`,
          `AO`, `AOPetscToApplication()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode DMDASetAOType(DM da, AOType aotype)
{
  DM_DA    *dd;
  PetscBool isdmda;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  PetscCall(PetscObjectTypeCompare((PetscObject)da, DMDA, &isdmda));
  PetscCheck(isdmda, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Requires a DMDA as input");
  /* now we can safely dereference */
  dd = (DM_DA *)da->data;
  if (dd->ao) { /* check if the already computed AO has the same type as requested */
    PetscBool match;
    PetscCall(PetscObjectTypeCompare((PetscObject)dd->ao, aotype, &match));
    PetscCheck(match, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Cannot change AO type");
    PetscFunctionReturn(0);
  }
  PetscCall(PetscFree(dd->aotype));
  PetscCall(PetscStrallocpy(aotype, (char **)&dd->aotype));
  PetscFunctionReturn(0);
}

/*@
   DMDAGetAO - Gets the application ordering context for a distributed array.

   Collective on da

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ao - the application ordering context for `DMDA`

   Level: intermediate

   Notes:
   In this case, the `AO` maps to the natural grid ordering that would be used
   for the `DMDA` if only 1 processor were employed (ordering most rapidly in the
   x-direction, then y, then z).  Multiple degrees of freedom are numbered
   for each node (rather than 1 component for the whole grid, then the next
   component, etc.)

   Do NOT call `AODestroy()` on the ao returned by this function.

.seealso: `DM`, `DMDA`, `DMDACreate2d()`, `DMDASetAOType()`, `DMDAGetGhostCorners()`, `DMDAGetCorners()`, `DMLocalToGlobal()`
          `DMGlobalToLocalBegin()`, `DMGlobalToLocalEnd()`, `DMLocalToLocalBegin()`, `DMLocalToLocalEnd()`, `DMDAGetOwnershipRanges()`,
          `AO`, `AOPetscToApplication()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode DMDAGetAO(DM da, AO *ao)
{
  DM_DA    *dd;
  PetscBool isdmda;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  PetscValidPointer(ao, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)da, DMDA, &isdmda));
  PetscCheck(isdmda, PetscObjectComm((PetscObject)da), PETSC_ERR_SUP, "Requires a DMDA as input");
  /* now we can safely dereference */
  dd = (DM_DA *)da->data;

  /*
     Build the natural ordering to PETSc ordering mappings.
  */
  if (!dd->ao) {
    IS       ispetsc, isnatural;
    PetscInt Nlocal;

    PetscCall(DMDAGetNatural_Private(da, &Nlocal, &isnatural));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)da), Nlocal, dd->base, 1, &ispetsc));
    PetscCall(AOCreate(PetscObjectComm((PetscObject)da), &dd->ao));
    PetscCall(AOSetIS(dd->ao, isnatural, ispetsc));
    PetscCall(AOSetType(dd->ao, dd->aotype));
    PetscCall(ISDestroy(&ispetsc));
    PetscCall(ISDestroy(&isnatural));
  }
  *ao = dd->ao;
  PetscFunctionReturn(0);
}
