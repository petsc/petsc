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
  static PetscErrorCode SolveInPlaceCyclic(PetscDeviceContext, PetscBool, PetscInt, PetscInt, const PetscScalar[], PetscInt, PetscScalar[], PetscInt) noexcept;
};

template <DeviceType T>
PetscErrorCode UpperTriangular<T>::SolveInPlace(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt N, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride) noexcept
{
  cupmBlasInt_t    n;
  cupmBlasHandle_t handle;
  auto             _A = cupmScalarPtrCast(A);
  auto             _x = cupmScalarPtrCast(x);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(N, &n));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n, _A, lda, _x, stride));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(1.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode UpperTriangular<T>::SolveInPlaceCyclic(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt N, PetscInt oldest_index, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride) noexcept
{
  cupmBlasInt_t         n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  auto                  sone      = cupmScalarCast(1.0);
  auto                  minus_one = cupmScalarCast(-1.0);
  auto                  _A        = cupmScalarPtrCast(A);
  auto                  _x        = cupmScalarPtrCast(x);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(N - oldest_index, &n_old));
  PetscCall(PetscCUPMBlasIntCast(oldest_index, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (!hermitian_transpose) {
    PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_new, _A, lda, _x, stride));
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, &minus_one, &_A[oldest_index], lda, _x, stride, &sone, &_x[oldest_index], stride));
    PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_old, &_A[oldest_index * (lda + 1)], lda, &_x[oldest_index], stride));
  } else {
    PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_old, &_A[oldest_index * (lda + 1)], lda, &_x[oldest_index], stride));
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, &minus_one, &_A[oldest_index], lda, &_x[oldest_index], stride, &sone, _x, stride));
    PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_new, _A, lda, _x, stride));
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

PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlaceCyclic_CUPM(PetscBool hermitian_transpose, PetscInt n, PetscInt oldest_index, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride)
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
    PetscCall(UpperTriangular<DeviceType::CUDA>::SolveInPlaceCyclic(dctx, hermitian_transpose, n, oldest_index, A, lda, x, stride));
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    PetscCall(UpperTriangular<DeviceType::HIP>::SolveInPlaceCyclic(dctx, hermitian_transpose, n, oldest_index, A, lda, x, stride));
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported device type %s", PetscDeviceTypes[device_type]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
