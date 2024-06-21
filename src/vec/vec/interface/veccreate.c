#include <petsc/private/vecimpl.h> /*I  "petscvec.h"   I*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>

static PetscErrorCode VecCreate_Common_Private(Vec v)
{
  PetscFunctionBegin;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecCreate - Creates an empty vector object. The type can then be set with `VecSetType()`,
  or `VecSetFromOptions().`

  Collective

  Input Parameter:
. comm - The communicator for the vector object

  Output Parameter:
. vec - The vector object

  Level: beginner

  Notes:
  If you never  call `VecSetType()` or `VecSetFromOptions()` it will generate an
  error when you try to use the vector.

.seealso: [](ch_vectors), `Vec`, `VecSetType()`, `VecSetSizes()`, `VecCreateMPIWithArray()`, `VecCreateMPI()`, `VecDuplicate()`,
          `VecDuplicateVecs()`, `VecCreateGhost()`, `VecCreateSeq()`, `VecPlaceArray()`
@*/
PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec)
{
  Vec v;

  PetscFunctionBegin;
  PetscAssertPointer(vec, 2);
  PetscCall(VecInitializePackage());

  PetscCall(PetscHeaderCreate(v, VEC_CLASSID, "Vec", "Vector", "Vec", comm, VecDestroy, VecView));
  PetscCall(PetscLayoutCreate(comm, &v->map));
  PetscCall(VecCreate_Common_Private(v));
  *vec = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecCreateFromOptions - Creates a vector whose type is set from the options database

  Collective

  Input Parameters:
+ comm   - The communicator for the vector object
. prefix - [optional] prefix for the options database
. bs     - the block size (commonly 1)
. m      - the local size (or `PETSC_DECIDE`)
- n      - the global size (or `PETSC_DETERMINE`)

  Output Parameter:
. vec - The vector object

  Options Database Keys:
. -vec_type - see `VecType`, for example `seq`, `mpi`, `cuda`, defaults to `mpi`

  Level: beginner

.seealso: [](ch_vectors), `Vec`, `VecSetType()`, `VecSetSizes()`, `VecCreateMPIWithArray()`, `VecCreateMPI()`, `VecDuplicate()`,
          `VecDuplicateVecs()`, `VecCreateGhost()`, `VecCreateSeq()`, `VecPlaceArray()`, `VecCreate()`, `VecType`
@*/
PetscErrorCode VecCreateFromOptions(MPI_Comm comm, const char *prefix, PetscInt bs, PetscInt m, PetscInt n, Vec *vec)
{
  PetscFunctionBegin;
  PetscAssertPointer(vec, 6);
  PetscCall(VecCreate(comm, vec));
  if (prefix) PetscCall(VecSetOptionsPrefix(*vec, prefix));
  PetscCall(VecSetBlockSize(*vec, bs));
  PetscCall(VecSetSizes(*vec, m, n));
  PetscCall(VecSetFromOptions(*vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create a vector with the given layout.  The reference count of the input layout will be increased by 1 */
PetscErrorCode VecCreateWithLayout_Private(PetscLayout map, Vec *vec)
{
  Vec v;

  PetscFunctionBegin;
  PetscAssertPointer(vec, 2);
  *vec = NULL;
  PetscCall(VecInitializePackage());
  PetscCall(PetscHeaderCreate(v, VEC_CLASSID, "Vec", "Vector", "Vec", map->comm, VecDestroy, VecView));
  v->map = map;
  map->refcnt++;
  PetscCall(VecCreate_Common_Private(v));
  v->bstash.bs = map->bs;
  *vec         = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}
