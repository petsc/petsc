#include "blas_cyclic_cupm.h"
#if PetscDefined(HAVE_CUPM)
  #include "blas_cyclic_cupm_impl.hpp"

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{
  #if PetscDefined(HAVE_CUDA)
template struct BLASCyclic<DeviceType::CUDA>;
  #endif

  #if PetscDefined(HAVE_HIP)
template struct BLASCyclic<DeviceType::HIP>;
  #endif
} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
#endif

#if !PetscDefined(HAVE_CUDA) && !PetscDefined(HAVE_HIP)
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wunused-parameter")
#endif

PETSC_INTERN PetscErrorCode AXPBYCyclic_CUPM_Private(PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar x[], PetscScalar beta, PetscScalar y[], PetscInt y_stride)
{
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::CUDA>::axpby(dctx, m, oldest, next, alpha, x, beta, y, y_stride));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::HIP>::axpby(dctx, m, oldest, next, alpha, x, beta, y, y_stride));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMVCyclic_CUPM_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], const PetscScalar x[], PetscScalar beta, PetscScalar y[])
{
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::CUDA>::dmv(dctx, hermitian_transpose, m, oldest, next, alpha, A, x, beta, y));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::HIP>::dmv(dctx, hermitian_transpose, m, oldest, next, alpha, A, x, beta, y));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DSVCyclic_CUPM_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], const PetscScalar x[], PetscScalar y[])
{
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::CUDA>::dsv(dctx, hermitian_transpose, m, oldest, next, A, x, y));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::HIP>::dsv(dctx, hermitian_transpose, m, oldest, next, A, x, y));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TRSVCyclic_CUPM_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar y[])
{
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::CUDA>::trsv(dctx, hermitian_transpose, m, oldest, next, A, lda, x, y));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::HIP>::trsv(dctx, hermitian_transpose, m, oldest, next, A, lda, x, y));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode HEMVCyclic_CUPM_Private(PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[])
{
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::CUDA>::hemv(dctx, m, oldest, next, alpha, A, lda, x, beta, y));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::HIP>::hemv(dctx, m, oldest, next, alpha, A, lda, x, beta, y));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode GEMVCyclic_CUPM_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[])
{
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::CUDA>::gemv(dctx, hermitian_transpose, m, oldest, next, alpha, A, lda, x, beta, y));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(::Petsc::device::cupm::impl::BLASCyclic<::Petsc::device::cupm::DeviceType::HIP>::gemv(dctx, hermitian_transpose, m, oldest, next, alpha, A, lda, x, beta, y));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !PetscDefined(HAVE_CUDA) && !PetscDefined(HAVE_HIP)
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()
#endif
