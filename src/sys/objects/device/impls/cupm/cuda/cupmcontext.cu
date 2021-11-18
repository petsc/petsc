#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  static const Petsc::CUPMContextCuda  contextCuda;
  PetscDeviceContext_(CUDA)           *dci;
  PetscErrorCode                       ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = static_cast<decltype(dctx->data)>(dci);
  ierr = PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Management of CUBLAS and CUSOLVER handles */
PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
