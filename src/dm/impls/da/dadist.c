
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_MPI_DA"
PetscErrorCode  VecDuplicate_MPI_DA(Vec g,Vec* gg)
{
  PetscErrorCode ierr;
  DM             da;

  PetscFunctionBegin; 
  ierr = PetscObjectQuery((PetscObject)g,"DM",(PetscObject*)&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,gg);CHKERRQ(ierr);
  ierr = PetscLayoutReference(g->map,&(*gg)->map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "DMCreateGlobalVector_DA"
PetscErrorCode  DMCreateGlobalVector_DA(DM da,Vec* g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(g,2);
  ierr = VecCreate(((PetscObject)da)->comm,g);CHKERRQ(ierr);
  ierr = VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*g,dd->w);CHKERRQ(ierr);
  ierr = VecSetType(*g,da->vectype);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*g);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*g,"DM",(PetscObject)da);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*g,da->ltogmap);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlock(*g,da->ltogmapb);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_VIEW,(void(*)(void))VecView_MPI_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_LOAD,(void(*)(void))VecLoad_Default_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_DUPLICATE,(void(*)(void))VecDuplicate_MPI_DA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDACreateNaturalVector"
/*@
   DMDACreateNaturalVector - Creates a parallel PETSc vector that
   will hold vector values in the natural numbering, rather than in 
   the PETSc parallel numbering associated with the DMDA.

   Collective on DMDA

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
          DMDACreate1d(), DMDACreate2d(), DMDACreate3d(), DMGlobalToLocalBegin(),
          DMGlobalToLocalEnd(), DMDALocalToGlobalBegin()
@*/
PetscErrorCode  DMDACreateNaturalVector(DM da,Vec* g)
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
    ierr = VecCreate(((PetscObject)da)->comm,g);CHKERRQ(ierr);
    ierr = VecSetSizes(*g,dd->Nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*g, dd->w);CHKERRQ(ierr);
    ierr = VecSetType(*g,VECMPI);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*g);CHKERRQ(ierr);
    dd->natural = *g;
  }
  PetscFunctionReturn(0);
}



