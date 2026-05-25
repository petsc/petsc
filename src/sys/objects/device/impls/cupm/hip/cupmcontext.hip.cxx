#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

using namespace Petsc::device::cupm;

PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext dctx)
{
  static constexpr auto hip_context = CUPMContextHip();

  PetscFunctionBegin;
  PetscCall(hip_context.initialize(dctx->device));
  dctx->data = new PetscDeviceContext_(HIP);
  *dctx->ops = hip_context.ops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Management of HIPBLAS and HIPSOLVER handles

 Unlike CUDA, hipSOLVER is just for dense matrices so there is
 no distinguishing being dense and sparse.  Also, hipSOLVER is
 very immature so we often have to do the mapping between roc and
 cuda manually.
 */

/*@C
  PetscHIPBLASGetHandle - Get the hipBLAS handle associated with PETSc's current `PetscDeviceContext`

  Not Collective; No Fortran Support

  Output Parameter:
. handle - the `hipblasHandle_t` for the current context

  Level: developer

  Note:
  The current device context must be of type `PETSC_DEVICE_HIP`. The returned handle is owned by
  PETSc and must not be destroyed by the caller.

.seealso: `PetscDeviceContext`, `PetscHIPSOLVERGetHandle()`, `PetscGetCurrentHIPStream()`
@*/
PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_HIP));
  PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscHIPSOLVERGetHandle - Get the hipSOLVER handle associated with PETSc's current `PetscDeviceContext`

  Not Collective; No Fortran Support

  Output Parameter:
. handle - the `hipsolverHandle_t` for the current context

  Level: developer

  Note:
  The current device context must be of type `PETSC_DEVICE_HIP`. The returned handle is owned by
  PETSc and must not be destroyed by the caller.

.seealso: `PetscDeviceContext`, `PetscHIPBLASGetHandle()`, `PetscGetCurrentHIPStream()`
@*/
PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t *handle)
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_HIP));
  PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscGetCurrentHIPStream - Get the HIP stream associated with PETSc's current `PetscDeviceContext`

  Not Collective; No Fortran Support

  Output Parameter:
. stream - the `hipStream_t` for the current context

  Level: developer

  Note:
  The current device context must be of type `PETSC_DEVICE_HIP`. The returned stream is owned by
  PETSc and must not be destroyed by the caller.

.seealso: `PetscDeviceContext`, `PetscHIPBLASGetHandle()`, `PetscHIPSOLVERGetHandle()`
@*/
PetscErrorCode PetscGetCurrentHIPStream(hipStream_t *stream)
{
  PetscDeviceContext dctx;
  void              *handle;

  PetscFunctionBegin;
  PetscAssertPointer(stream, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, PETSC_DEVICE_HIP));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &handle));
  *stream = *(hipStream_t *)handle;
  PetscFunctionReturn(PETSC_SUCCESS);
}
