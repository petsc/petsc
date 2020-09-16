
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

PetscErrorCode  VecDuplicate_MPI_DA(Vec g,Vec *gg)
{
  PetscErrorCode ierr;
  DM             da;
  PetscLayout    map;

  PetscFunctionBegin;
  ierr = VecGetDM(g, &da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,gg);CHKERRQ(ierr);
  ierr = VecGetLayout(g,&map);CHKERRQ(ierr);
  ierr = VecSetLayout(*gg,map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateGlobalVector_DA(DM da,Vec *g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(g,2);
  ierr = VecCreate(PetscObjectComm((PetscObject)da),g);CHKERRQ(ierr);
  ierr = VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*g,dd->w);CHKERRQ(ierr);
  ierr = VecSetType(*g,da->vectype);CHKERRQ(ierr);
  if (dd->Nlocal < da->bind_below) {ierr = VecBindToCPU(*g,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = VecSetDM(*g, da);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*g,da->ltogmap);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_VIEW,(void (*)(void))VecView_MPI_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_LOAD,(void (*)(void))VecLoad_Default_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_DUPLICATE,(void (*)(void))VecDuplicate_MPI_DA);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       cnt;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidPointer(g,2);
  if (dd->natural) {
    ierr = PetscObjectGetReference((PetscObject)dd->natural,&cnt);CHKERRQ(ierr);
    if (cnt == 1) { /* object is not currently used by anyone */
      ierr = PetscObjectReference((PetscObject)dd->natural);CHKERRQ(ierr);
      *g   = dd->natural;
    } else {
      ierr = VecDuplicate(dd->natural,g);CHKERRQ(ierr);
    }
  } else { /* create the first version of this guy */
    ierr = VecCreate(PetscObjectComm((PetscObject)da),g);CHKERRQ(ierr);
    ierr = VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*g, dd->w);CHKERRQ(ierr);
    ierr = VecSetType(*g,da->vectype);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*g);CHKERRQ(ierr);

    dd->natural = *g;
  }
  PetscFunctionReturn(0);
}

