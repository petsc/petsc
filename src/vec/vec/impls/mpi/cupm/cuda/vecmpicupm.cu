#include "../vecmpicupm.hpp" /*I <petscvec.h> I*/

using namespace Petsc::vec::cupm::impl;

static constexpr auto VecMPI_CUDA = VecMPI_CUPM<::Petsc::device::cupm::DeviceType::CUDA>{};

/*MC
  VECCUDA - VECCUDA = "cuda" - A VECSEQCUDA on a single-process communicator, and VECMPICUDA
  otherwise.

  Options Database Keys:
. -vec_type cuda - sets the vector type to VECCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQCUDA,
VECMPICUDA, VECSTANDARD, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

/*MC
  VECMPICUDA - VECMPICUDA = "mpicuda" - The basic parallel vector, modified to use CUDA

  Options Database Keys:
. -vec_type mpicuda - sets the vector type to VECMPICUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI,
VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecCreate_CUDA(Vec v)
{
  PetscFunctionBegin;
  PetscCall(VecMPI_CUDA.Create_CUPM(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecCreate_MPICUDA(Vec v)
{
  PetscFunctionBegin;
  PetscCall(VecMPI_CUDA.create(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecCUDAGetArrays_Private(Vec v, const PetscScalar **host_array, const PetscScalar **device_array, PetscOffloadMask *mask)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(VecMPI_CUDA.GetArrays_CUPMBase(v, host_array, device_array, mask, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecCreateMPICUDA - Creates a standard, parallel, array-style vector for CUDA devices.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm - the MPI communicator to use
. n    - local vector length (or PETSC_DECIDE to have calculated if N is given)
- N    - global vector length (or PETSC_DETERMINE to have calculated if n is given)

  Output Parameter:
. v - the vector

  Notes:
  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an
  existing vector.

  This function may initialize PetscDevice, which may incur a device synchronization.

  Level: intermediate

.seealso: VecCreateMPICUDAWithArray(), VecCreateMPICUDAWithArrays(), VecCreateSeqCUDA(),
VecCreateSeq(), VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
VecCreateGhost(), VecCreateMPIWithArray(), VecCreateGhostWithArray(), VecMPISetGhost()
@*/
PetscErrorCode VecCreateMPICUDA(MPI_Comm comm, PetscInt n, PetscInt N, Vec *v)
{
  PetscFunctionBegin;
  PetscValidPointer(v, 4);
  PetscCall(VecMPI_CUDA.creatempicupm(comm, 0, n, N, v, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecCreateMPICUDAWithArrays - Creates a parallel, array-style vector using CUDA, where the
  user provides the complete array space to store the vector values.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm     - the MPI communicator to use
. bs       - block size, same meaning as VecSetBlockSize()
. n        - local vector length, cannot be PETSC_DECIDE
. N        - global vector length (or PETSC_DECIDE to have calculated)
. cpuarray - CPU memory where the vector elements are to be stored (or NULL)
- gpuarray - GPU memory where the vector elements are to be stored (or NULL)

  Output Parameter:
. v - the vector

  Notes:
  See VecCreateSeqCUDAWithArrays() for further discussion, this routine shares identical
  semantics.

  Level: intermediate

.seealso: VecCreateMPICUDA(), VecCreateSeqCUDAWithArrays(), VecCreateMPIWithArray(),
VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()
@*/
PetscErrorCode VecCreateMPICUDAWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  if (n && cpuarray) PetscValidScalarPointer(cpuarray, 5);
  PetscValidPointer(v, 7);
  PetscCall(VecMPI_CUDA.creatempicupmwitharrays(comm, bs, n, N, cpuarray, gpuarray, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecCreateMPICUDAWithArray - Creates a parallel, array-style vector using CUDA, where the
  user provides the device array space to store the vector values.
  Collective

  Input Parameters:
+ comm  - the MPI communicator to use
. bs    - block size, same meaning as VecSetBlockSize()
. n     - local vector length, cannot be PETSC_DECIDE
. N     - global vector length (or PETSC_DECIDE to have calculated)
- gpuarray - the user provided GPU array to store the vector values

  Output Parameter:
. v - the vector

  Notes:
  See VecCreateSeqCUDAWithArray() for further discussion, this routine shares identical
  semantics.

  Level: intermediate

.seealso: VecCreateMPICUDA(), VecCreateSeqCUDAWithArray(), VecCreateMPIWithArray(),
VecCreateSeqWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()
@*/
PetscErrorCode VecCreateMPICUDAWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecCreateMPICUDAWithArrays(comm, bs, n, N, nullptr, gpuarray, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
