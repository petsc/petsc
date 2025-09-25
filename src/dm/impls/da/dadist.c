/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

PetscErrorCode DMCreateGlobalVector_DA(DM da, Vec *g)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID, 1);
  PetscAssertPointer(g, 2);
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da), g));
  PetscCall(VecSetSizes(*g, dd->Nlocal, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(*g, dd->w));
  PetscCall(VecSetType(*g, da->vectype));
  if (dd->Nlocal < da->bind_below) {
    PetscCall(VecSetBindingPropagates(*g, PETSC_TRUE));
    PetscCall(VecBindToCPU(*g, PETSC_TRUE));
  }
  PetscCall(VecSetDM(*g, da));
  PetscCall(VecSetLocalToGlobalMapping(*g, da->ltogmap));
  PetscCall(VecSetOperation(*g, VECOP_VIEW, (PetscErrorCodeFn *)VecView_MPI_DA));
  PetscCall(VecSetOperation(*g, VECOP_LOAD, (PetscErrorCodeFn *)VecLoad_Default_DA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDACreateNaturalVector - Creates a parallel PETSc vector that
  will hold vector values in the natural numbering, rather than in
  the PETSc parallel numbering associated with the `DMDA`.

  Collective

  Input Parameter:
. da - the `DMDA`

  Output Parameter:
. g - the distributed global vector

  Level: advanced

  Notes:
  The natural numbering is a number of grid nodes that starts with, in three dimensions, with (0,0,0), (1,0,0), (2,0,0), ..., (m-1,0,0) followed by
  (0,1,0), (1,1,0), (2,1,0), ..., (m,1,0) etc up to (0,n-1,p-1), (1,n-1,p-1), (2,n-1,p-1), ..., (m-1,n-1,p-1).

  The output parameter, `g`, is a regular `Vec` that should be destroyed
  with a call to `VecDestroy()` when usage is finished.

  The number of local entries in the vector on each process is the same
  as in a vector created with `DMCreateGlobalVector()`.

.seealso: [](sec_struct), `DM`, `DMDA`, `DMDAGlobalToNaturalBegin()`, `DMDAGlobalToNaturalEnd()`, `DMDANaturalToGlobalBegin()`, `DMDANaturalToGlobalEnd()`,
          `DMCreateLocalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToLocalBegin()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`
@*/
PetscErrorCode DMDACreateNaturalVector(DM da, Vec *g)
{
  PetscInt cnt;
  DM_DA   *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da, DM_CLASSID, 1, DMDA);
  PetscAssertPointer(g, 2);
  if (dd->natural) {
    PetscCall(PetscObjectGetReference((PetscObject)dd->natural, &cnt));
    if (cnt == 1) { /* object is not currently used by anyone */
      PetscCall(PetscObjectReference((PetscObject)dd->natural));
      *g = dd->natural;
    } else PetscCall(VecDuplicate(dd->natural, g));
  } else { /* create the first version of this guy */
    PetscCall(VecCreate(PetscObjectComm((PetscObject)da), g));
    PetscCall(VecSetSizes(*g, dd->Nlocal, PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(*g, dd->w));
    PetscCall(VecSetType(*g, da->vectype));
    PetscCall(PetscObjectReference((PetscObject)*g));
    dd->natural = *g;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
