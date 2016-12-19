
#include <petsc/private/vecimpl.h>           /*I  "petscvec.h"   I*/

/*@
  VecCreate - Creates an empty vector object. The type can then be set with VecSetType(),
  or VecSetFromOptions().

   If you never  call VecSetType() or VecSetFromOptions() it will generate an
   error when you try to use the vector.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the vector object

  Output Parameter:
. vec  - The vector object

  Level: beginner

.keywords: vector, create
.seealso: VecSetType(), VecSetSizes(), VecCreateMPIWithArray(), VecCreateMPI(), VecDuplicate(),
          VecDuplicateVecs(), VecCreateGhost(), VecCreateSeq(), VecPlaceArray()
@*/
PetscErrorCode  VecCreate(MPI_Comm comm, Vec *vec)
{
  Vec            v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(vec,2);
  *vec = NULL;
  ierr = VecInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(v, VEC_CLASSID, "Vec", "Vector", "Vec", comm, VecDestroy, VecView);CHKERRQ(ierr);

  ierr            = PetscLayoutCreate(comm,&v->map);CHKERRQ(ierr);
  v->array_gotten = PETSC_FALSE;
  v->petscnative  = PETSC_FALSE;

  *vec = v;
  PetscFunctionReturn(0);
}

