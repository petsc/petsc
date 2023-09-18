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
PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
