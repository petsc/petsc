#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DMCreateGlobalVector_DA"
PetscErrorCode PETSCDM_DLLEXPORT DMCreateGlobalVector_DA(DM da,Vec* g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(g,2);
  ierr = VecCreate(((PetscObject)da)->comm,g);CHKERRQ(ierr);
  ierr = VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(*g,da->vectype);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*g,dd->ltogmap);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlock(*g,dd->ltogmapb);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*g,dd->w);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_VIEW,(void(*)(void))VecView_MPI_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_LOAD,(void(*)(void))VecLoad_Default_DA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DACreateNaturalVector"
/*@
   DACreateNaturalVector - Creates a parallel PETSc vector that
   will hold vector values in the natural numbering, rather than in 
   the PETSc parallel numbering associated with the DA.

   Collective on DA

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

.keywords: distributed array, create, global, distributed, vector

.seealso: DMCreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DALocalToGlobalBegin()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreateNaturalVector(DM da,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt       cnt;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
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
    ierr = VecCreateMPI(((PetscObject)da)->comm,dd->Nlocal,PETSC_DETERMINE,g);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*g, dd->w);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*g);CHKERRQ(ierr);
    dd->natural = *g;
  }
  PetscFunctionReturn(0);
}



