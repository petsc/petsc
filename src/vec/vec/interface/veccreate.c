
#include <petsc/private/vecimpl.h> /*I  "petscvec.h"   I*/

/*@
  VecCreate - Creates an empty vector object. The type can then be set with `VecSetType()`,
  or `VecSetFromOptions().`

  Collective

  Input Parameter:
. comm - The communicator for the vector object

  Output Parameter:
. vec  - The vector object

  Level: beginner

  Notes:
   If you never  call `VecSetType()` or `VecSetFromOptions()` it will generate an
   error when you try to use the vector.

.seealso: [](chapter_vectors), `Vec`, `VecSetType()`, `VecSetSizes()`, `VecCreateMPIWithArray()`, `VecCreateMPI()`, `VecDuplicate()`,
          `VecDuplicateVecs()`, `VecCreateGhost()`, `VecCreateSeq()`, `VecPlaceArray()`
@*/
PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec)
{
  Vec v;

  PetscFunctionBegin;
  PetscValidPointer(vec, 2);
  *vec = NULL;
  PetscCall(VecInitializePackage());

  PetscCall(PetscHeaderCreate(v, VEC_CLASSID, "Vec", "Vector", "Vec", comm, VecDestroy, VecView));

  PetscCall(PetscLayoutCreate(comm, &v->map));
  v->array_gotten = PETSC_FALSE;
  v->petscnative  = PETSC_FALSE;
  v->offloadmask  = PETSC_OFFLOAD_UNALLOCATED;
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  v->minimum_bytes_pinned_memory = 0;
  v->pinned_memory               = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_DEVICE)
  v->boundtocpu = PETSC_TRUE;
#endif
  PetscCall(PetscStrallocpy(PETSCRANDER48, &v->defaultrandtype));
  *vec = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}
