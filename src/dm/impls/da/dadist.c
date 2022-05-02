
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

PetscErrorCode  VecDuplicate_MPI_DA(Vec g,Vec *gg)
{
  DM             da;
  PetscLayout    map;

  PetscFunctionBegin;
  PetscCall(VecGetDM(g, &da));
  PetscCall(DMCreateGlobalVector(da,gg));
  PetscCall(VecGetLayout(g,&map));
  PetscCall(VecSetLayout(*gg,map));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateGlobalVector_DA(DM da,Vec *g)
{
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(g,2);
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da),g));
  PetscCall(VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(*g,dd->w));
  PetscCall(VecSetType(*g,da->vectype));
  if (dd->Nlocal < da->bind_below) {
    PetscCall(VecSetBindingPropagates(*g,PETSC_TRUE));
    PetscCall(VecBindToCPU(*g,PETSC_TRUE));
  }
  PetscCall(VecSetDM(*g, da));
  PetscCall(VecSetLocalToGlobalMapping(*g,da->ltogmap));
  PetscCall(VecSetOperation(*g,VECOP_VIEW,(void (*)(void))VecView_MPI_DA));
  PetscCall(VecSetOperation(*g,VECOP_LOAD,(void (*)(void))VecLoad_Default_DA));
  PetscCall(VecSetOperation(*g,VECOP_DUPLICATE,(void (*)(void))VecDuplicate_MPI_DA));
  PetscFunctionReturn(0);
}

/*@
   DMDACreateNaturalVector - Creates a parallel PETSc vector that
   will hold vector values in the natural numbering, rather than in
   the PETSc parallel numbering associated with the DMDA.

   Collective

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed global vector

   Level: developer

   Note:
   The output parameter, g, is a regular PETSc vector that should be destroyed
   with a call to VecDestroy() when usage is finished.

   The number of local entries in the vector on each process is the same
   as in a vector created with DMCreateGlobalVector().

.seealso: `DMCreateLocalVector()`, `VecDuplicate()`, `VecDuplicateVecs()`,
          `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToLocalBegin()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`
@*/
PetscErrorCode  DMDACreateNaturalVector(DM da,Vec *g)
{
  PetscInt       cnt;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidPointer(g,2);
  if (dd->natural) {
    PetscCall(PetscObjectGetReference((PetscObject)dd->natural,&cnt));
    if (cnt == 1) { /* object is not currently used by anyone */
      PetscCall(PetscObjectReference((PetscObject)dd->natural));
      *g   = dd->natural;
    } else {
      PetscCall(VecDuplicate(dd->natural,g));
    }
  } else { /* create the first version of this guy */
    PetscCall(VecCreate(PetscObjectComm((PetscObject)da),g));
    PetscCall(VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE));
    PetscCall(VecSetBlockSize(*g, dd->w));
    PetscCall(VecSetType(*g,da->vectype));
    PetscCall(PetscObjectReference((PetscObject)*g));

    dd->natural = *g;
  }
  PetscFunctionReturn(0);
}
