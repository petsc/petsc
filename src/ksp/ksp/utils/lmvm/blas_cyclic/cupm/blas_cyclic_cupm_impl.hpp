#pragma once
#include "blas_cyclic_cupm.h"
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
struct BLASCyclic : CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  static PetscErrorCode axpby_dispatch(cupmBlasHandle_t, cupmBlasInt_t, PetscScalar, const PetscScalar[], PetscScalar, PetscScalar[], cupmBlasInt_t) noexcept;
  static PetscErrorCode axpby(PetscDeviceContext, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscScalar, PetscScalar[], PetscInt) noexcept;
  static PetscErrorCode dmv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], const PetscScalar[], PetscScalar, PetscScalar[]) noexcept;
  static PetscErrorCode dsv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], PetscScalar[]) noexcept;
  static PetscErrorCode trsv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[]) noexcept;
  static PetscErrorCode gemv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar, PetscScalar[]) noexcept;
  static PetscErrorCode hemv(PetscDeviceContext, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar, PetscScalar[]) noexcept;
};

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::axpby_dispatch(cupmBlasHandle_t handle, cupmBlasInt_t n, PetscScalar alpha, const PetscScalar x[], PetscScalar beta, PetscScalar y[], cupmBlasInt_t y_stride) noexcept
{
  auto       x_     = cupmScalarPtrCast(x);
  auto       y_     = cupmScalarPtrCast(y);
  const auto calpha = cupmScalarPtrCast(&alpha);
  const auto cbeta  = cupmScalarPtrCast(&beta);

  PetscFunctionBegin;
  if (alpha == 1.0 && beta == 0.0) {
    PetscCallCUPMBLAS(cupmBlasXcopy(handle, n, x_, 1, y_, y_stride));
  } else {
    if (beta != 1.0) PetscCallCUPMBLAS(cupmBlasXscal(handle, n, cbeta, y_, y_stride));
    if (alpha != 0.0) PetscCallCUPMBLAS(cupmBlasXaxpy(handle, n, calpha, x_, 1, y_, y_stride));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::axpby(PetscDeviceContext dctx, PetscInt M, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar x[], PetscScalar beta, PetscScalar y[], PetscInt y_stride) noexcept
{
  PetscInt              N = next - oldest;
  cupmBlasInt_t         m, i_oldest, i_next, y_stride_;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(M, &m));
  PetscCall(PetscCUPMBlasIntCast(oldest % m, &i_oldest));
  PetscCall(PetscCUPMBlasIntCast(((next - 1) % m) + 1, &i_next));
  PetscCall(PetscCUPMBlasIntCast(y_stride, &y_stride_));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (N == m) {
    PetscCall(axpby_dispatch(handle, m, alpha, x, beta, y, y_stride_));
  } else if (i_next > i_oldest) {
    cupmBlasInt_t diff = i_next - i_oldest;

    PetscCall(axpby_dispatch(handle, diff, alpha, &x[i_oldest], beta, &y[i_oldest * y_stride], y_stride_));
  } else {
    cupmBlasInt_t diff = m - i_oldest;

    if (i_next) PetscCall(axpby_dispatch(handle, i_next, alpha, x, beta, y, y_stride_));
    if (diff) PetscCall(axpby_dispatch(handle, diff, alpha, &x[i_oldest], beta, &y[i_oldest * y_stride], y_stride_));
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(3.0 * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::dmv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt M, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], const PetscScalar x[], PetscScalar beta, PetscScalar y[]) noexcept
{
  PetscInt              N = next - oldest;
  cupmBlasInt_t         m, i_oldest, i_next;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  const auto            A_     = cupmScalarPtrCast(A);
  const auto            x_     = cupmScalarPtrCast(x);
  const auto            y_     = cupmScalarPtrCast(y);
  const auto            calpha = cupmScalarPtrCast(&alpha);
  const auto            cbeta  = cupmScalarPtrCast(&beta);
  const auto            trans  = hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N;

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(M, &m));
  PetscCall(PetscCUPMBlasIntCast(oldest % m, &i_oldest));
  PetscCall(PetscCUPMBlasIntCast(((next - 1) % m) + 1, &i_next));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (N == m) {
    PetscCallCUPMBLAS(cupmBlasXgbmv(handle, trans, m, m, 0, 0, calpha, A_, 1, x_, 1, cbeta, y_, 1));
  } else if (i_next > i_oldest) {
    cupmBlasInt_t diff = i_next - i_oldest;

    PetscCallCUPMBLAS(cupmBlasXgbmv(handle, trans, diff, diff, 0, 0, calpha, &A_[i_oldest], 1, &x_[i_oldest], 1, cbeta, &y_[i_oldest], 1));
  } else {
    cupmBlasInt_t diff = m - i_oldest;

    if (i_next) PetscCallCUPMBLAS(cupmBlasXgbmv(handle, trans, i_next, i_next, 0, 0, calpha, A_, 1, x_, 1, cbeta, y_, 1));
    if (diff) PetscCallCUPMBLAS(cupmBlasXgbmv(handle, trans, diff, diff, 0, 0, calpha, &A_[i_oldest], 1, &x_[i_oldest], 1, cbeta, &y_[i_oldest], 1));
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(3.0 * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::dsv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt M, PetscInt oldest, PetscInt next, const PetscScalar A[], const PetscScalar x[], PetscScalar y[]) noexcept
{
  PetscInt              N = next - oldest;
  cupmBlasInt_t         m, i_oldest, i_next;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  const auto            A_    = cupmScalarPtrCast(A);
  const auto            y_    = cupmScalarPtrCast(y);
  auto                  trans = hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N;

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(M, &m));
  PetscCall(PetscCUPMBlasIntCast(oldest % m, &i_oldest));
  PetscCall(PetscCUPMBlasIntCast(((next - 1) % m) + 1, &i_next));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (N == m) {
    if (x != y) PetscCall(PetscCUPMMemcpyAsync(y, x, m, cupmMemcpyDeviceToDevice, stream));
    PetscCallCUPMBLAS(cupmBlasXtbsv(handle, CUPMBLAS_FILL_MODE_UPPER, trans, CUPMBLAS_DIAG_NON_UNIT, m, 0, A_, 1, y_, 1));
  } else if (i_next > i_oldest) {
    cupmBlasInt_t diff = i_next - i_oldest;

    if (x != y) PetscCall(PetscCUPMMemcpyAsync(&y[i_oldest], &x[i_oldest], diff, cupmMemcpyDeviceToDevice, stream));
    PetscCallCUPMBLAS(cupmBlasXtbsv(handle, CUPMBLAS_FILL_MODE_UPPER, trans, CUPMBLAS_DIAG_NON_UNIT, diff, 0, &A_[i_oldest], 1, &y_[i_oldest], 1));
  } else {
    cupmBlasInt_t diff = m - i_oldest;

    if (i_next) {
      if (x != y) PetscCall(PetscCUPMMemcpyAsync(y, x, i_next, cupmMemcpyDeviceToDevice, stream));
      PetscCallCUPMBLAS(cupmBlasXtbsv(handle, CUPMBLAS_FILL_MODE_UPPER, trans, CUPMBLAS_DIAG_NON_UNIT, i_next, 0, A_, 1, y_, 1));
    }
    if (diff) {
      if (x != y) PetscCall(PetscCUPMMemcpyAsync(&y[i_oldest], &x[i_oldest], diff, cupmMemcpyDeviceToDevice, stream));
      PetscCallCUPMBLAS(cupmBlasXtbsv(handle, CUPMBLAS_FILL_MODE_UPPER, trans, CUPMBLAS_DIAG_NON_UNIT, diff, 0, &A_[i_oldest], 1, &y_[i_oldest], 1));
    }
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(3.0 * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::trsv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar y[]) noexcept
{
  PetscInt              N        = next - oldest;
  PetscInt              i_oldest = oldest % m;
  PetscInt              i_next   = ((next - 1) % m) + 1;
  cupmBlasInt_t         n, n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  auto                  sone      = cupmScalarCast(1.0);
  auto                  minus_one = cupmScalarCast(-1.0);
  auto                  A_        = cupmScalarPtrCast(A);
  auto                  y_        = cupmScalarPtrCast(y);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(i_next - i_oldest, &n));
  PetscCall(PetscCUPMBlasIntCast(m - i_oldest, &n_old));
  PetscCall(PetscCUPMBlasIntCast(i_next, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (n > 0) {
    if (x != y) PetscCall(PetscCUPMMemcpyAsync(y, x, n, cupmMemcpyDeviceToDevice, stream));
    PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n, &A_[i_oldest * (lda + 1)], lda, y_, 1));
  } else if (!hermitian_transpose) {
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_new, A_, lda, y_, 1));
    if (n_new > 0 && n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, &minus_one, &A_[i_oldest], lda, y_, 1, &sone, &y_[i_oldest], 1));
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_old, &A_[i_oldest * (lda + 1)], lda, &y_[i_oldest], 1));
  } else {
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_old, &A_[i_oldest * (lda + 1)], lda, &y_[i_oldest], 1));
    if (n_new > 0 && n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, &minus_one, &A_[i_oldest], lda, &y_[i_oldest], 1, &sone, y_, 1));
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_new, A_, lda, y_, 1));
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(1.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::hemv(PetscDeviceContext dctx, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[]) noexcept
{
  PetscInt              N        = next - oldest;
  PetscInt              i_oldest = oldest % m;
  PetscInt              i_next   = ((next - 1) % m) + 1;
  cupmBlasInt_t         n, n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  auto                  sone   = cupmScalarCast(1.0);
  auto                  A_     = cupmScalarPtrCast(A);
  auto                  x_     = cupmScalarPtrCast(x);
  auto                  y_     = cupmScalarPtrCast(y);
  const auto            calpha = cupmScalarPtrCast(&alpha);
  const auto            cbeta  = cupmScalarPtrCast(&beta);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(i_next - i_oldest, &n));
  PetscCall(PetscCUPMBlasIntCast(m - i_oldest, &n_old));
  PetscCall(PetscCUPMBlasIntCast(i_next, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (n > 0) {
    PetscCallCUPMBLAS(cupmBlasXhemv(handle, CUPMBLAS_FILL_MODE_UPPER, n, calpha, &A_[i_oldest * (lda + 1)], lda, &x_[i_oldest], 1, cbeta, &y_[i_oldest], 1));
  } else {
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXhemv(handle, CUPMBLAS_FILL_MODE_UPPER, n_new, calpha, A_, lda, x_, 1, cbeta, y_, 1));
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXhemv(handle, CUPMBLAS_FILL_MODE_UPPER, n_old, calpha, &A_[i_oldest * (lda + 1)], lda, &x_[i_oldest], 1, cbeta, &y_[i_oldest], 1));
    if (n_new > 0 && n_old > 0) {
      PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, calpha, &A_[i_oldest], lda, x_, 1, &sone, &y_[i_oldest], 1));
      PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, calpha, &A_[i_oldest], lda, &x_[i_oldest], 1, &sone, y_, 1));
    }
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(2.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::gemv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[]) noexcept
{
  PetscInt              N        = next - oldest;
  PetscInt              i_oldest = oldest % m;
  PetscInt              i_next   = ((next - 1) % m) + 1;
  cupmBlasInt_t         n, n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  auto                  sone   = cupmScalarCast(1.0);
  auto                  A_     = cupmScalarPtrCast(A);
  auto                  x_     = cupmScalarPtrCast(x);
  auto                  y_     = cupmScalarPtrCast(y);
  auto                  trans  = hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N;
  const auto            calpha = cupmScalarPtrCast(&alpha);
  const auto            cbeta  = cupmScalarPtrCast(&beta);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(i_next - i_oldest, &n));
  PetscCall(PetscCUPMBlasIntCast(m - i_oldest, &n_old));
  PetscCall(PetscCUPMBlasIntCast(i_next, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (N == m) {
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, N, N, calpha, A_, lda, x_, 1, cbeta, y_, 1));
  } else if (n > 0) {
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, n, n, calpha, &A_[i_oldest * (lda + 1)], lda, &x_[i_oldest], 1, cbeta, &y_[i_oldest], 1));
  } else {
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, n_new, n_new, calpha, A_, lda, x_, 1, cbeta, y_, 1));
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, n_old, n_old, calpha, &A_[i_oldest * (lda + 1)], lda, &x_[i_oldest], 1, cbeta, &y_[i_oldest], 1));
    if (n_new > 0 && n_old > 0) {
      if (!hermitian_transpose) {
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, calpha, &A_[i_oldest], lda, x_, 1, &sone, &y_[i_oldest], 1));
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_new, n_old, calpha, &A_[i_oldest * lda], lda, &x_[i_oldest], 1, &sone, y_, 1));
      } else {
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_new, n_old, calpha, &A_[i_oldest * lda], lda, x_, 1, &sone, &y_[i_oldest], 1));
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, calpha, &A_[i_oldest], lda, &x_[i_oldest], 1, &sone, y_, 1));
      }
    }
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(2.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
