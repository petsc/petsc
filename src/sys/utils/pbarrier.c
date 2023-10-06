#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/

/* Logging support */
PetscLogEvent PETSC_Barrier;

/*@C
  PetscBarrier - Blocks until this routine is executed by all processors owning the object `obj`.

  Input Parameter:
. obj - PETSc object  (`Mat`, `Vec`, `IS`, `SNES` etc...)

  Level: intermediate

  Notes:
  The object must be cast with a (`PetscObject`). `NULL` can be used to indicate the barrier
  should be across `PETSC_COMM_WORLD`.

  Developer Notes:
  This routine calls `MPI_Barrier()` with the communicator of the `PetscObject`

  Fortran Notes:
  You may pass `PETSC_NULL_VEC` or any other PETSc null object, such as `PETSC_NULL_MAT`, to
  indicate the barrier should be across `PETSC_COMM_WORLD`. You can also pass in any PETSc
  object, `Vec`, `Mat`, etc.

.seealso: `PetscObject`
@*/
PetscErrorCode PetscBarrier(PetscObject obj)
{
  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  if (obj) {
    PetscValidHeader(obj, 1);
    PetscCall(PetscObjectGetComm(obj, &comm));
  }
  PetscCall(PetscLogEventBegin(PETSC_Barrier, obj, 0, 0, 0));
  PetscCallMPI(MPI_Barrier(comm));
  PetscCall(PetscLogEventEnd(PETSC_Barrier, obj, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
