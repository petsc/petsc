
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

PetscErrorCode  VecDuplicate_MPI_DA(Vec g,Vec *gg)
{
  DM             da;
  PetscLayout    map;

  PetscFunctionBegin;
  CHKERRQ(VecGetDM(g, &da));
  CHKERRQ(DMCreateGlobalVector(da,gg));
  CHKERRQ(VecGetLayout(g,&map));
  CHKERRQ(VecSetLayout(*gg,map));
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateGlobalVector_DA(DM da,Vec *g)
{
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(g,2);
  CHKERRQ(VecCreate(PetscObjectComm((PetscObject)da),g));
  CHKERRQ(VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE));
  CHKERRQ(VecSetBlockSize(*g,dd->w));
  CHKERRQ(VecSetType(*g,da->vectype));
  if (dd->Nlocal < da->bind_below) {
    CHKERRQ(VecSetBindingPropagates(*g,PETSC_TRUE));
    CHKERRQ(VecBindToCPU(*g,PETSC_TRUE));
  }
  CHKERRQ(VecSetDM(*g, da));
  CHKERRQ(VecSetLocalToGlobalMapping(*g,da->ltogmap));
  CHKERRQ(VecSetOperation(*g,VECOP_VIEW,(void (*)(void))VecView_MPI_DA));
  CHKERRQ(VecSetOperation(*g,VECOP_LOAD,(void (*)(void))VecLoad_Default_DA));
  CHKERRQ(VecSetOperation(*g,VECOP_DUPLICATE,(void (*)(void))VecDuplicate_MPI_DA));
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

.seealso: DMCreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMLocalToGlobalBegin()
@*/
PetscErrorCode  DMDACreateNaturalVector(DM da,Vec *g)
{
  PetscInt       cnt;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidPointer(g,2);
  if (dd->natural) {
    CHKERRQ(PetscObjectGetReference((PetscObject)dd->natural,&cnt));
    if (cnt == 1) { /* object is not currently used by anyone */
      CHKERRQ(PetscObjectReference((PetscObject)dd->natural));
      *g   = dd->natural;
    } else {
      CHKERRQ(VecDuplicate(dd->natural,g));
    }
  } else { /* create the first version of this guy */
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)da),g));
    CHKERRQ(VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE));
    CHKERRQ(VecSetBlockSize(*g, dd->w));
    CHKERRQ(VecSetType(*g,da->vectype));
    CHKERRQ(PetscObjectReference((PetscObject)*g));

    dd->natural = *g;
  }
  PetscFunctionReturn(0);
}
