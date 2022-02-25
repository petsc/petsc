#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

using namespace Petsc::Device::CUPM;

PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext dctx)
{
  static constexpr auto     contextHip = CUPMContextHip();
  PetscDeviceContext_(HIP) *dci;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = static_cast<decltype(dctx->data)>(dci);
  ierr = PetscMemcpy(dctx->ops,&contextHip.ops,sizeof(contextHip.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Management of HIPBLAS and HIPSOLVER handles

 Unlike CUDA, hipSOLVER is just for dense matrices so there is
 no distinguishing being dense and sparse.  Also, hipSOLVER is
 very immature so we often have to do the mapping between roc and
 cuda manually.
 */

PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_HIP);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t *handle)
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,PETSC_DEVICE_HIP);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
