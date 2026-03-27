#pragma once

#include <petscmat.h>

#include "../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp"

#include <petsc/private/cupminterface.hpp>
#include <petsc/private/cupmobject.hpp>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/veccupmimpl.h>
#include <petsc/private/matimpl.h>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

template <DeviceType T, typename VecType>
struct MatDiagonal_CUPM : vec::cupm::impl::Vec_CUPMBase<T, VecType> {
  PETSC_CUPMOBJECT_HEADER(T);
  using base_type = ::Petsc::vec::cupm::impl::Vec_CUPMBase<T, VecType>;
  friend base_type;

  static PetscErrorCode ADot(Mat A, Vec x, Vec y, PetscScalar *z) noexcept;
  static PetscErrorCode ANormSq(Mat A, Vec x, PetscReal *z) noexcept;
};

namespace detail
{
struct adot_transform {
  using argument_type = thrust::tuple<PetscScalar, PetscScalar, PetscScalar>;

  PETSC_NODISCARD PETSC_HOSTDEVICE_INLINE_DECL PetscScalar operator()(const argument_type &tup) const noexcept { return PetscConj(thrust::get<1>(tup)) * thrust::get<2>(tup) * thrust::get<0>(tup); }
};
} // namespace detail

template <Petsc::device::cupm::DeviceType T, typename VecType>
inline PetscErrorCode MatDiagonal_CUPM<T, VecType>::ADot(Mat A, Vec x, Vec y, PetscScalar *z) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  Mat_Diagonal      *ctx  = (Mat_Diagonal *)A->data;
  PetscScalar        zero = 0.;
  const PetscInt     n    = x->map->n;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));

  const auto xdptr = thrust::device_pointer_cast(base_type::DeviceArrayRead(dctx, x).data());
  const auto ydptr = thrust::device_pointer_cast(base_type::DeviceArrayRead(dctx, y).data());
  const auto wdptr = thrust::device_pointer_cast(base_type::DeviceArrayRead(dctx, ctx->diag).data());

  // clang-format off
    PetscCallThrust(
      *z = THRUST_CALL(
        thrust::transform_reduce,
        stream,
        thrust::make_zip_iterator(thrust::make_tuple(xdptr, ydptr, wdptr)),
        thrust::make_zip_iterator(thrust::make_tuple(xdptr + n, ydptr + n, wdptr + n)),
        detail::adot_transform{},
        zero,
        thrust::plus<PetscScalar>()
      )
    );
  // clang-format on
  if (x->map->n > 0) PetscCall(PetscLogGpuFlops(3.0 * x->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace detail
{
struct anorm_transform {
  using argument_type = thrust::tuple<PetscScalar, PetscScalar>;

  PETSC_NODISCARD PETSC_HOSTDEVICE_INLINE_DECL PetscScalar operator()(const argument_type &tup) const noexcept { return thrust::get<1>(tup) * PetscConj(thrust::get<0>(tup)) * thrust::get<0>(tup); }
};
} // namespace detail

template <Petsc::device::cupm::DeviceType T, typename VecType>
inline PetscErrorCode MatDiagonal_CUPM<T, VecType>::ANormSq(Mat A, Vec x, PetscReal *z) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  Mat_Diagonal      *ctx  = (Mat_Diagonal *)A->data;
  PetscScalar        zero = 0., res;
  const PetscInt     n    = x->map->n;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));

  const auto xdptr = thrust::device_pointer_cast(base_type::DeviceArrayRead(dctx, x).data());
  const auto wdptr = thrust::device_pointer_cast(base_type::DeviceArrayRead(dctx, ctx->diag).data());

  // clang-format off
  PetscCallThrust(
    res = THRUST_CALL(
      thrust::transform_reduce,
      stream,
      thrust::make_zip_iterator(thrust::make_tuple(xdptr, wdptr)),
      thrust::make_zip_iterator(thrust::make_tuple(xdptr + n, wdptr + n)),
      detail::anorm_transform{},
      zero,
      thrust::plus<PetscScalar>()
    )
  );
  // clang-format on
  *z = PetscRealPart(res);
  if (x->map->n > 0) PetscCall(PetscLogGpuFlops(3.0 * x->map->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
