
#include "vecimpl.h"      /*I "petscvec.h"  I*/


#undef __FUNCT__  
#define __FUNCT__ "PetscMapCreate"
/*@C
  PetscMapCreate - Creates an empty map object. The type can then be set with PetscMapSetType().

  Collective on MPI_Comm
 
  Input Parameter:
. comm - The MPI communicator for the map object 

  Output Parameter:
. map  - The map object

  Level: beginner

.keywords: PetscMap, create
.seealso: PetscMapDestroy(), PetscMapGetLocalSize(), PetscMapGetSize(), PetscMapGetGlobalRange(), PetscMapGetLocalRange()
@*/ 
PetscErrorCode PetscMapCreate(MPI_Comm comm, PetscMap *map)
{
  PetscMap m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(map,2);
  *map = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  PetscHeaderCreate(m, _p_PetscMap, struct _PetscMapOps, MAP_COOKIE, -1, "PetscMap", comm, PetscMapDestroy, PETSC_NULL);
  PetscLogObjectMemory(m, sizeof(struct _p_PetscMap));
  ierr = PetscMemzero(m->ops, sizeof(struct _PetscMapOps));CHKERRQ(ierr);
  m->bops->publish  = PETSC_NULL /* PetscMapPublish_Petsc */;
  m->type_name      = PETSC_NULL;

  m->n      = -1;
  m->N      = -1;
  m->rstart = -1;
  m->rend   = -1;
  m->range  = PETSC_NULL;

  *map = m;
  PetscFunctionReturn(0);
}

