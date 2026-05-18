#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

using namespace Petsc::device::cupm;

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  static constexpr auto cuda_context = CUPMContextCuda();

  PetscFunctionBegin;
  PetscCall(cuda_context.initialize(dctx->device));
  dctx->data = new PetscDeviceContext_(CUDA);
  *dctx->ops = cuda_context.ops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Management of CUBLAS and CUSOLVER handles */
/*@C
  PetscCUBLASGetHandle - Get the cuBLAS handle associated with PETSc's current `PetscDeviceContext`

  Not Collective; No Fortran Support

  Output Parameter:
. handle - the `cublasHandle_t` for the current context

  Level: developer

  Note:
  The current device context must be of type `PETSC_DEVICE_CUDA`. The returned handle is owned by
  PETSc and must not be destroyed by the caller.

.seealso: `PetscDeviceContext`, `PetscDeviceContextSetUp()`, `PetscCUSOLVERDnGetHandle()`, `PetscGetCurrentCUDAStream()`
@*/
PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscCUSOLVERDnGetHandle - Get the cuSolverDn handle associated with PETSc's current `PetscDeviceContext`

  Not Collective; No Fortran Support

  Output Parameter:
. handle - the `cusolverDnHandle_t` for the current context

  Level: developer

  Note:
  The current device context must be of type `PETSC_DEVICE_CUDA`. The returned handle is owned by
  PETSc and must not be destroyed by the caller.

.seealso: `PetscDeviceContext`, `PetscCUBLASGetHandle()`, `PetscGetCurrentCUDAStream()`
@*/
PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscGetCurrentCUDAStream - Get the CUDA stream associated with PETSc's current `PetscDeviceContext`

  Not Collective; No Fortran Support

  Output Parameter:
. stream - the `cudaStream_t` for the current context

  Level: developer

  Note:
  The current device context must be of type `PETSC_DEVICE_CUDA`. The returned stream is owned by
  PETSc and must not be destroyed by the caller.

.seealso: `PetscDeviceContext`, `PetscCUBLASGetHandle()`, `PetscCUSOLVERDnGetHandle()`
@*/
PetscErrorCode PetscGetCurrentCUDAStream(cudaStream_t *stream)
{
  PetscDeviceContext dctx;
  void              *handle;

  PetscFunctionBegin;
  PetscAssertPointer(stream, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &handle));
  *stream = *(cudaStream_t *)handle;
  PetscFunctionReturn(PETSC_SUCCESS);
}
