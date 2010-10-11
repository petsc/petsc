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

   When you view this vector (or one obtained via VecDuplicate()) it is printed in the global natural ordering NOT 
   in the PETSc parallel global ordering that is used internally. Similarly VecLoad() into this vector loads from a global natural ordering. 
   This means that vectors saved to disk from one DA parallel distribution can be reloaded into a different DA parallel distribution correctly.

.keywords: distributed array, create, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal(), DACreateNaturalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreateGlobalVector(DA da,Vec* g)
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
   as in a vector created with DACreateGlobalVector().

.keywords: distributed array, create, global, distributed, vector

.seealso: DACreateLocalVector(), VecDuplicate(), VecDuplicateVecs(),
          DACreate1d(), DACreate2d(), DACreate3d(), DAGlobalToLocalBegin(),
          DAGlobalToLocalEnd(), DALocalToGlobal()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreateNaturalVector(DA da,Vec* g)
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



