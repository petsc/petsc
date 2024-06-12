#include <petscmat.h> /*I    "petscmat.h"   I*/

/*@
  MatCreateDenseFromVecType - Create a matrix that matches the type of a Vec.

  Collective

  Input Parameters:
+ comm  - the communicator
. vtype - the vector type
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n     - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M     - number of global rows (or `PETSC_DECIDE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DECIDE` to have calculated if `n` is given)
. lda   - optional leading dimension. Pass any non-positive number to use the default.
- data  - optional location of matrix data, which should have the same memory type as the vector. Pass `NULL` to have PETSc take care of matrix memory allocation.

  Output Parameter:
. A - the dense matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateDense()`, `MatCreateDenseCUDA()`, `MatCreateDenseHIP()`, `PetscMemType`
@*/
PetscErrorCode MatCreateDenseFromVecType(MPI_Comm comm, VecType vtype, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt lda, PetscScalar *data, Mat *A)
{
  VecType   root_type = VECSTANDARD;
  PetscBool isstd, iscuda, iship, iskokkos;

  PetscFunctionBegin;
  PetscCall(PetscStrcmpAny(vtype, &isstd, VECSTANDARD, VECMPI, VECSEQ, ""));
  PetscCall(PetscStrcmpAny(vtype, &iscuda, VECCUDA, VECMPICUDA, VECSEQCUDA, ""));
  PetscCall(PetscStrcmpAny(vtype, &iship, VECHIP, VECMPIHIP, VECSEQHIP, ""));
  PetscCall(PetscStrcmpAny(vtype, &iskokkos, VECKOKKOS, VECMPIKOKKOS, VECSEQKOKKOS, ""));
  PetscCheck(isstd || iscuda || iship || iskokkos, comm, PETSC_ERR_SUP, "Not for type %s", vtype);
  if (iscuda) root_type = VECCUDA;
  else if (iship) root_type = VECHIP;
  else if (iskokkos) {
    /* We support only one type of kokkos device */
    PetscCheck(!PetscDefined(HAVE_MACRO_KOKKOS_ENABLE_SYCL), comm, PETSC_ERR_SUP, "Not for sycl backend");
    if (PetscDefined(HAVE_MACRO_KOKKOS_ENABLE_CUDA)) iscuda = PETSC_TRUE;
    else if (PetscDefined(HAVE_MACRO_KOKKOS_ENABLE_HIP)) iship = PETSC_TRUE;
    else isstd = PETSC_TRUE;
    root_type = VECKOKKOS;
  }
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  if (isstd) {
    PetscCall(MatSetType(*A, MATDENSE));
    if (lda > 0) PetscCall(MatDenseSetLDA(*A, lda));
    PetscCall(MatSeqDenseSetPreallocation(*A, data));
    PetscCall(MatMPIDenseSetPreallocation(*A, data));
  } else if (iscuda) {
    PetscCheck(PetscDefined(HAVE_CUDA), comm, PETSC_ERR_SUP, "PETSc not compiled with CUDA support");
#if defined(PETSC_HAVE_CUDA)
    PetscCall(MatSetType(*A, MATDENSECUDA));
    if (lda > 0) PetscCall(MatDenseSetLDA(*A, lda));
    PetscCall(MatDenseCUDASetPreallocation(*A, data));
#endif
  } else if (iship) {
    PetscCheck(PetscDefined(HAVE_HIP), comm, PETSC_ERR_SUP, "PETSc not compiled with HIP support");
#if defined(PETSC_HAVE_HIP)
    PetscCall(MatSetType(*A, MATDENSEHIP));
    if (lda > 0) PetscCall(MatDenseSetLDA(*A, lda));
    PetscCall(MatDenseHIPSetPreallocation(*A, data));
#endif
  }
  PetscCall(MatSetVecType(*A, root_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}
