#include <petscmat.h>              /*I "petscmat.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

/*@
  VecCreateMatDense - Create a matrix that matches the type of a Vec.

  Collective

  Input Parameters:
+ X    - the vector
. m    - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n    - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M    - number of global rows (or `PETSC_DECIDE` to have calculated if `m` is given)
. N    - number of global columns (or `PETSC_DECIDE` to have calculated if `n` is given)
- data - optional location of matrix data, which should have the same memory type as the vector. Pass `NULL` to have PETSc to control matrix.
         memory allocation.

  Output Parameter:
. A - the matrix.  `A` will have the same communicator as `X` and the same `PetscMemType`.

  Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MatCreateDense()', `MatCreateDenseCUDA()`, `MatCreateDenseHIP()`, `PetscMemType`
@*/
PetscErrorCode VecCreateMatDense(Vec X, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar *data, Mat *A)
{
  VecType   root_type;
  PetscBool isstd, iscuda, iship;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(VecGetRootType_Private(X, &root_type));
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCall(PetscStrcmp(root_type, VECSTANDARD, &isstd));
  PetscCall(PetscStrcmp(root_type, VECCUDA, &iscuda));
  PetscCall(PetscStrcmp(root_type, VECHIP, &iship));

  /* For performance-portable types (Kokkos, SYCL, ...) that dispatch to */
  if (!(isstd || iscuda || iship)) {
    const PetscScalar *array;
    PetscMemType       memtype;

    PetscCall(VecGetArrayReadAndMemType(X, &array, &memtype));
    PetscCall(VecRestoreArrayReadAndMemType(X, &array));
    switch (memtype) {
    case PETSC_MEMTYPE_HOST:
      isstd = PETSC_TRUE;
      break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
      iscuda = PETSC_TRUE;
      break;
    case PETSC_MEMTYPE_HIP:
      iship = PETSC_TRUE;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)X), PETSC_ERR_SUP, "Cannot figure out memory type of vector type %s", root_type);
    }
  }

  if (isstd) {
    PetscCall(MatCreateDense(comm, m, n, M, N, data, A));
  }
#if defined(PETSC_HAVE_CUDA)
  else if (iscuda) {
    PetscCall(MatCreateDenseCUDA(comm, m, n, M, N, data, A));
  }
#endif
#if defined(PETSC_HAVE_HIP)
  else if (iship) {
    PetscCall(MatCreateDenseHIP(comm, m, n, M, N, data, A));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
