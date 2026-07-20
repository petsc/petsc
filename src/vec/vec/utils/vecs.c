#include <petscvec.h>

/*@
  VecsDestroy - Destroys a `Vecs` collection of vectors

  Collective

  Input Parameter:
. x - the `Vecs` object to destroy

  Level: advanced

.seealso: `Vecs`, `VecsCreateSeq()`, `VecsCreateSeqWithArray()`, `VecsDuplicate()`
@*/
PetscErrorCode VecsDestroy(Vecs x)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&(x)->v));
  PetscCall(PetscFree(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecsCreateSeq - Creates a `Vecs` object holding `p` sequential `Vec`s of length `m`, all stored contiguously in a single underlying `Vec`

  Collective

  Input Parameters:
+ comm - the MPI communicator (typically `PETSC_COMM_SELF`)
. p    - the number of vectors
- m    - the length of each vector

  Output Parameter:
. x - the newly created `Vecs`

  Level: advanced

.seealso: `Vecs`, `VecsCreateSeqWithArray()`, `VecsDuplicate()`, `VecsDestroy()`, `VecCreateSeq()`
@*/
PetscErrorCode VecsCreateSeq(MPI_Comm comm, PetscInt p, PetscInt m, Vecs *x)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(x));
  PetscCall(VecCreateSeq(comm, p * m, &(*x)->v));
  (*x)->n = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecsCreateSeqWithArray - Creates a `Vecs` object holding `p` sequential `Vec`s of length `m` that use a user-provided contiguous array as storage

  Collective

  Input Parameters:
+ comm - the MPI communicator (typically `PETSC_COMM_SELF`)
. p    - the number of vectors
. m    - the length of each vector
- a    - the array of length `p*m` used as storage for the vectors

  Output Parameter:
. x - the newly created `Vecs`

  Level: advanced

.seealso: `Vecs`, `VecsCreateSeq()`, `VecsDuplicate()`, `VecsDestroy()`, `VecCreateSeqWithArray()`
@*/
PetscErrorCode VecsCreateSeqWithArray(MPI_Comm comm, PetscInt p, PetscInt m, PetscScalar *a, Vecs *x)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(x));
  PetscCall(VecCreateSeqWithArray(comm, 1, p * m, a, &(*x)->v));
  (*x)->n = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecsDuplicate - Creates a new `Vecs` with the same size and layout as an existing `Vecs`, but does not copy the values

  Collective

  Input Parameter:
. x - the existing `Vecs`

  Output Parameter:
. y - the newly created `Vecs`

  Level: advanced

.seealso: `Vecs`, `VecsCreateSeq()`, `VecsCreateSeqWithArray()`, `VecsDestroy()`, `VecDuplicate()`
@*/
PetscErrorCode VecsDuplicate(Vecs x, Vecs *y)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(y));
  PetscCall(VecDuplicate(x->v, &(*y)->v));
  (*y)->n = x->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}
