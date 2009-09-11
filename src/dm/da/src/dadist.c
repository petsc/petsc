#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DACreateGlobalVector"
/*@
   DACreateGlobalVector - Creates a parallel PETSc vector that
   may be used with the DAXXX routines.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameter:
.  g - the distributed global vector

   Level: beginner

   Note:
   The output parameter, g, is a regular PETSc vector that should be destroyed
   with a call to VecDestroy() when usage is finished.

.keywords: distributed array, create, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DACreateNaturalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreateGlobalVector(DA da,Vec* g)
{
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(g,2);
  ierr = VecCreateMPI(((PetscObject)da)->comm,da->Nlocal,PETSC_DETERMINE,g);
  ierr = PetscObjectCompose((PetscObject)*g,"DA",(PetscObject)da);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*g,da->ltogmap);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlock(*g,da->ltogmapb);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*g,da->w);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_VIEW,(void(*)(void))VecView_MPI_DA);CHKERRQ(ierr);
  ierr = VecSetOperation(*g,VECOP_LOADINTOVECTOR,(void(*)(void))VecLoadIntoVector_Binary_DA);CHKERRQ(ierr);
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
   as in a vector created with DACreateGlobalVector().

.keywords: distributed array, create, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreateNaturalVector(DA da,Vec* g)
{
  PetscErrorCode ierr;
  PetscInt cnt;

  PetscFunctionBegin; 
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(g,2);
  if (da->natural) {
    ierr = PetscObjectGetReference((PetscObject)da->natural,&cnt);CHKERRQ(ierr);
    if (cnt == 1) { /* object is not currently used by anyone */
      ierr = PetscObjectReference((PetscObject)da->natural);CHKERRQ(ierr);
      *g   = da->natural;
    } else {
      ierr = VecDuplicate(da->natural,g);CHKERRQ(ierr);
    }
  } else { /* create the first version of this guy */
    ierr = VecCreateMPI(((PetscObject)da)->comm,da->Nlocal,PETSC_DETERMINE,g);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*g, da->w);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)*g);CHKERRQ(ierr);
    da->natural = *g;
  }
  PetscFunctionReturn(0);
}



