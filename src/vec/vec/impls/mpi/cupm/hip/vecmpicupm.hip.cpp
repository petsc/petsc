#include "../vecmpicupm.hpp" /*I <petscvec.h> I*/

using namespace Petsc::vec::cupm;
using Petsc::device::cupm::DeviceType;

static constexpr auto VecMPI_HIP = impl::VecMPI_CUPM<DeviceType::HIP>{};

/*MC
  VECHIP - VECHIP = "hip" - A `VECSEQHIP` on a single-process communicator, and `VECMPIHIP`
  otherwise.

  Options Database Keys:
. -vec_type hip - sets the vector type to `VECHIP` during a call to `VecSetFromOptions()`

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateMPIWithArray()`, `VECSEQHIP`,
`VECMPIHIP`, `VECSTANDARD`, `VecType`, `VecCreateMPI()`, `VecSetPinnedMemoryMin()`
M*/

/*MC
  VECMPIHIP - VECMPIHIP = "mpihip" - The basic parallel vector, modified to use HIP

  Options Database Keys:
. -vec_type mpihip - sets the vector type to `VECMPIHIP` during a call to `VecSetFromOptions()`

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateMPIWithArray()`, `VECMPI`,
`VecType`, `VecCreateMPI()`, `VecSetPinnedMemoryMin()`
M*/

PetscErrorCode VecCreate_HIP(Vec v)
{
  PetscFunctionBegin;
  PetscCall(VecMPI_HIP.Create_CUPM(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecCreate_MPIHIP(Vec v)
{
  PetscFunctionBegin;
  PetscCall(VecMPI_HIP.Create(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecHIPGetArrays_Private(Vec v, const PetscScalar **host_array, const PetscScalar **device_array, PetscOffloadMask *mask)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_HIP));
  PetscCall(VecMPI_HIP.GetArrays_CUPMBase(v, host_array, device_array, mask, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecCreateMPIHIP - Creates a standard, parallel, array-style vector for HIP devices.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm - the MPI communicator to use
. n    - local vector length (or `PETSC_DECIDE` to have calculated if N is given)
- N    - global vector length (or `PETSC_DETERMINE` to have calculated if n is given)

  Output Parameter:
. v - the vector

  Notes:
  Use `VecDuplicate()` or `VecDuplicateVecs()` to form additional vectors of the same type as an
  existing vector.

  This function may initialize `PetscDevice`, which may incur a device synchronization.

  Level: intermediate

.seealso: `VecCreateMPIHIPWithArray()`, `VecCreateMPIHIPWithArrays()`, `VecCreateSeqHIP()`,
`VecCreateSeq()`, `VecCreateMPI()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`,
`VecCreateGhost()`, `VecCreateMPIWithArray()`, `VecCreateGhostWithArray()`, `VecMPISetGhost()`
@*/
PetscErrorCode VecCreateMPIHIP(MPI_Comm comm, PetscInt n, PetscInt N, Vec *v)
{
  PetscFunctionBegin;
  PetscValidPointer(v, 4);
  PetscCall(VecCreateMPICUPMAsync<DeviceType::HIP>(comm, n, N, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecCreateMPIHIPWithArrays - Creates a parallel, array-style vector using HIP, where the
  user provides the complete array space to store the vector values.

  Collective, Possibly Synchronous

  Input Parameters:
+ comm     - the MPI communicator to use
. bs       - block size, same meaning as `VecSetBlockSize()`
. n        - local vector length, cannot be `PETSC_DECIDE`
. N        - global vector length (or `PETSC_DECIDE` to have calculated)
. cpuarray - CPU memory where the vector elements are to be stored (or `NULL`)
- gpuarray - GPU memory where the vector elements are to be stored (or `NULL`)

  Output Parameter:
. v - the vector

  Notes:
  See `VecCreateSeqHIPWithArrays()` for further discussion, this routine shares identical
  semantics.

  Level: intermediate

.seealso: `VecCreateMPIHIP()`, `VecCreateSeqHIPWithArrays()`, `VecCreateMPIWithArray()`,
`VecCreateSeqWithArray()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`,
`VecCreateMPI()`, `VecCreateGhostWithArray()`, `VecPlaceArray()`
@*/
PetscErrorCode VecCreateMPIHIPWithArrays(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar cpuarray[], const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecCreateMPICUPMWithArrays<DeviceType::HIP>(comm, bs, n, N, cpuarray, gpuarray, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecCreateMPIHIPWithArray - Creates a parallel, array-style vector using HIP, where the
  user provides the device array space to store the vector values.

  Collective

  Input Parameters:
+ comm  - the MPI communicator to use
. bs    - block size, same meaning as `VecSetBlockSize()`
. n     - local vector length, cannot be `PETSC_DECIDE`
. N     - global vector length (or `PETSC_DECIDE` to have calculated)
- gpuarray - the user provided GPU array to store the vector values

  Output Parameter:
. v - the vector

  Notes:
  See `VecCreateSeqHIPWithArray()` for further discussion, this routine shares identical
  semantics.

  Level: intermediate

.seealso: `VecCreateMPIHIP()`, `VecCreateSeqHIPWithArray()`, `VecCreateMPIWithArray()`,
`VecCreateSeqWithArray()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`, `VecCreateGhost()`,
`VecCreateMPI()`, `VecCreateGhostWithArray()`, `VecPlaceArray()`
@*/
PetscErrorCode VecCreateMPIHIPWithArray(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar gpuarray[], Vec *v)
{
  PetscFunctionBegin;
  PetscCall(VecCreateMPICUPMWithArray<DeviceType::HIP>(comm, bs, n, N, gpuarray, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
