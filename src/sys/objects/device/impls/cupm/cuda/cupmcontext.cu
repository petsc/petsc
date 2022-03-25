#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

using namespace Petsc::Device::CUPM;

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  static constexpr auto     contextCuda = CUPMContextCuda();
  PetscDeviceContext_(CUDA) *dci;

  PetscFunctionBegin;
  PetscCall(PetscNew(&dci));
  dctx->data = static_cast<decltype(dctx->data)>(dci);
  PetscCall(PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops)));
  PetscFunctionReturn(0);
}

/* Management of CUBLAS and CUSOLVER handles */
PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx,handle));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_CUDA));
  PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle));
  PetscFunctionReturn(0);
}
