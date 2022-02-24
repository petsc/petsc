#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

using namespace Petsc::Device::CUPM;

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  static constexpr auto     contextCuda = CUPMContextCuda();
  PetscDeviceContext_(CUDA) *dci;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&dci));
  dctx->data = static_cast<decltype(dctx->data)>(dci);
  CHKERRQ(PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops)));
  PetscFunctionReturn(0);
}

/* Management of CUBLAS and CUSOLVER handles */
PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  CHKERRQ(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_CUDA));
  CHKERRQ(PetscDeviceContextGetBLASHandle_Internal(dctx,handle));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  CHKERRQ(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_CUDA));
  CHKERRQ(PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle));
  PetscFunctionReturn(0);
}
