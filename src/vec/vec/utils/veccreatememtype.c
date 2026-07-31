#include <petscvec.h> /*I "petscvec.h" I*/
#include <petsc/private/petscimpl.h>

/*@
  VecCreateSeqWithArrayAndMemType - Creates a sequential array-style vector using a user-provided array with the specified memory type

  Collective

  Input Parameters:
+ comm  - the communicator, should be `PETSC_COMM_SELF`
. mtype - the memory type of `array`, which must be the same on all processes in `comm`
. bs    - the block size
. n     - the vector length
- array - memory where the vector elements are to be stored

  Output Parameter:
. V - the vector

  Level: intermediate

  Notes:
  `mtype` determines whether the resulting vector is a standard, CUDA, or HIP vector. Since `PETSC_MEMTYPE_DEVICE` and
  `PETSC_MEMTYPE_CUDA` have the same value, that value creates a CUDA vector when CUDA is configured and otherwise creates
  a HIP vector when HIP is configured.

  `mtype` alone does not identify the Kokkos implementation, and native SYCL vectors are not supported, so this function does
  not create Kokkos or SYCL vectors.

  `array` remains owned by the caller and is not freed when the vector is destroyed via `VecDestroy()`.

.seealso: `VecCreateSeqWithArray()`, `VecCreateMPIWithArrayAndMemType()`, `VecCreateSeqCUDAWithArray()`, `VecCreateSeqHIPWithArray()`, `PetscMemType`
@*/
PetscErrorCode VecCreateSeqWithArrayAndMemType(MPI_Comm comm, PetscMemType mtype, PetscInt bs, PetscInt n, const PetscScalar array[], Vec *V)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(comm, (PetscInt)mtype, 2);
  if (mtype == PETSC_MEMTYPE_DEVICE) {
    PetscCheck(PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP), comm, PETSC_ERR_SUP, "Not for PetscMemType %s without CUDA or HIP support", PetscMemTypeToString(mtype));
    if (PetscDefined(HAVE_CUDA)) PetscCall(VecCreateSeqCUDAWithArray(comm, bs, n, array, V));
    else PetscCall(VecCreateSeqHIPWithArray(comm, bs, n, array, V));
  } else if (PetscMemTypeCUDA(mtype)) {
    PetscCheck(PetscDefined(HAVE_CUDA), comm, PETSC_ERR_SUP, "Not for PetscMemType %s without CUDA support", PetscMemTypeToString(mtype));
    PetscCall(VecCreateSeqCUDAWithArray(comm, bs, n, array, V));
  } else if (PetscMemTypeHIP(mtype)) {
    PetscCheck(PetscDefined(HAVE_HIP), comm, PETSC_ERR_SUP, "Not for PetscMemType %s without HIP support", PetscMemTypeToString(mtype));
    PetscCall(VecCreateSeqHIPWithArray(comm, bs, n, array, V));
  } else {
    PetscCheck(PetscMemTypeHost(mtype), comm, PETSC_ERR_SUP, "Not for PetscMemType %s", PetscMemTypeToString(mtype));
    PetscCall(VecCreateSeqWithArray(comm, bs, n, array, V));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecCreateMPIWithArrayAndMemType - Creates a parallel array-style vector using a user-provided array with the specified memory type

  Collective

  Input Parameters:
+ comm  - the MPI communicator to use
. mtype - the memory type of `array`, which must be the same on all processes in `comm`
. bs    - block size, same meaning as `VecSetBlockSize()`
. n     - local vector length, cannot be `PETSC_DECIDE`
. N     - global vector length (or `PETSC_DETERMINE` to have calculated)
- array - memory where the vector elements are to be stored

  Output Parameter:
. V - the vector

  Level: intermediate

  Notes:
  `mtype` determines whether the resulting vector is a standard, CUDA, or HIP vector. Since `PETSC_MEMTYPE_DEVICE` and
  `PETSC_MEMTYPE_CUDA` have the same value, that value creates a CUDA vector when CUDA is configured and otherwise creates
  a HIP vector when HIP is configured.

  `mtype` alone does not identify the Kokkos implementation, and native SYCL vectors are not supported, so this function does
  not create Kokkos or SYCL vectors.

  `array` remains owned by the caller and is not freed when the vector is destroyed via `VecDestroy()`.

.seealso: `VecCreateMPIWithArray()`, `VecCreateSeqWithArrayAndMemType()`, `VecCreateMPICUDAWithArray()`, `VecCreateMPIHIPWithArray()`, `PetscMemType`
@*/
PetscErrorCode VecCreateMPIWithArrayAndMemType(MPI_Comm comm, PetscMemType mtype, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar array[], Vec *V)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(comm, (PetscInt)mtype, 2);
  if (mtype == PETSC_MEMTYPE_DEVICE) {
    PetscCheck(PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP), comm, PETSC_ERR_SUP, "Not for PetscMemType %s without CUDA or HIP support", PetscMemTypeToString(mtype));
    if (PetscDefined(HAVE_CUDA)) PetscCall(VecCreateMPICUDAWithArray(comm, bs, n, N, array, V));
    else PetscCall(VecCreateMPIHIPWithArray(comm, bs, n, N, array, V));
  } else if (PetscMemTypeCUDA(mtype)) {
    PetscCheck(PetscDefined(HAVE_CUDA), comm, PETSC_ERR_SUP, "Not for PetscMemType %s without CUDA support", PetscMemTypeToString(mtype));
    PetscCall(VecCreateMPICUDAWithArray(comm, bs, n, N, array, V));
  } else if (PetscMemTypeHIP(mtype)) {
    PetscCheck(PetscDefined(HAVE_HIP), comm, PETSC_ERR_SUP, "Not for PetscMemType %s without HIP support", PetscMemTypeToString(mtype));
    PetscCall(VecCreateMPIHIPWithArray(comm, bs, n, N, array, V));
  } else {
    PetscCheck(PetscMemTypeHost(mtype), comm, PETSC_ERR_SUP, "Not for PetscMemType %s", PetscMemTypeToString(mtype));
    PetscCall(VecCreateMPIWithArray(comm, bs, n, N, array, V));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
