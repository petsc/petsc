
/*
   Implements the sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h> /*I  "petscvec.h"   I*/

/*@
   VecCreateSeq - Creates a standard, sequential array-style vector.

   Collective

   Input Parameters:
+  comm - the communicator, should be `PETSC_COMM_SELF`
-  n - the vector length

   Output Parameter:
.  V - the vector

   Level: intermediate

   Notes:
   Use `VecDuplicate()` or `VecDuplicateVecs()` to form additional vectors of the
   same type as an existing vector.

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreateMPI()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`
@*/
PetscErrorCode VecCreateSeq(MPI_Comm comm, PetscInt n, Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecCreate(comm, v));
  PetscCall(VecSetSizes(*v, n, n));
  PetscCall(VecSetType(*v, VECSEQ));
  PetscFunctionReturn(PETSC_SUCCESS);
}
