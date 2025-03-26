#include "../denseqn.h"
#include <petsc/private/cupminterface.hpp>
#include <petsc/private/cupmobject.hpp>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

template <DeviceType T>
struct UpperTriangular : CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  static PetscErrorCode SolveInPlace(PetscDeviceContext, PetscBool, PetscInt, const PetscScalar[], PetscInt, PetscScalar[], PetscInt) noexcept;
  static PetscErrorCode SolveInPlaceCyclic(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscInt, PetscScalar[], PetscInt) noexcept;
};

template <DeviceType T>
PetscErrorCode UpperTriangular<T>::SolveInPlace(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt N, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride) noexcept
{
  cupmBlasInt_t    n;
  cupmBlasHandle_t handle;
  auto             A_ = cupmScalarPtrCast(A);
  auto             x_ = cupmScalarPtrCast(x);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(N, &n));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n, A_, lda, x_, stride));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(1.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode UpperTriangular<T>::SolveInPlaceCyclic(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride) noexcept
{
  PetscInt              N            = next - oldest;
  PetscInt              oldest_index = oldest % m;
  PetscInt              next_index   = next % m;
  cupmBlasInt_t         n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  auto                  sone      = cupmScalarCast(1.0);
  auto                  minus_one = cupmScalarCast(-1.0);
  auto                  A_        = cupmScalarPtrCast(A);
  auto                  x_        = cupmScalarPtrCast(x);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(m - oldest_index, &n_old));
  PetscCall(PetscCUPMBlasIntCast(next_index, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (!hermitian_transpose) {
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_new, A_, lda, x_, stride));
    if (n_new > 0 && n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, &minus_one, &A_[oldest_index], lda, x_, stride, &sone, &x_[oldest_index], stride));
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_old, &A_[oldest_index * (lda + 1)], lda, &x_[oldest_index], stride));
  } else {
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_old, &A_[oldest_index * (lda + 1)], lda, &x_[oldest_index], stride));
    if (n_new > 0 && n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, &minus_one, &A_[oldest_index], lda, &x_[oldest_index], stride, &sone, x_, stride));
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_new, A_, lda, x_, stride));
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(1.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(HAVE_CUDA)
template struct UpperTriangular<DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
template struct UpperTriangular<DeviceType::HIP>;
#endif

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlace_CUPM(PetscBool hermitian_transpose, PetscInt n, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride)
{
  using ::Petsc::device::cupm::impl::UpperTriangular;
  using ::Petsc::device::cupm::DeviceType;
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(UpperTriangular<DeviceType::CUDA>::SolveInPlace(dctx, hermitian_transpose, n, A, lda, x, stride));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(UpperTriangular<DeviceType::HIP>::SolveInPlace(dctx, hermitian_transpose, n, A, lda, x, stride));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlaceCyclic_CUPM(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride)
{
  using ::Petsc::device::cupm::impl::UpperTriangular;
  using ::Petsc::device::cupm::DeviceType;
  PetscDeviceContext dctx;
  PetscDeviceType    device_type;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  switch (device_type) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    PetscCall(UpperTriangular<DeviceType::CUDA>::SolveInPlaceCyclic(dctx, hermitian_transpose, m, oldest, next, A, lda, x, stride));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(UpperTriangular<DeviceType::HIP>::SolveInPlaceCyclic(dctx, hermitian_transpose, m, oldest, next, A, lda, x, stride));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
