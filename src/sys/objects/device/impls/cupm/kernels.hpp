#ifndef PETSC_DEVICE_CUPM_KERNELS_HPP
#define PETSC_DEVICE_CUPM_KERNELS_HPP

#include <petscdevice_cupm.h>

#if defined(__cplusplus)

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace kernels
{

namespace util
{

template <typename SizeType, typename T>
PETSC_DEVICE_INLINE_DECL static void grid_stride_1D(const SizeType size, T &&func) noexcept
{
  for (SizeType i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) func(i);
  return;
}

} // namespace util

} // namespace kernels

namespace functors
{

template <typename T>
class plus_equals {
public:
  using value_type = T;

  PETSC_HOSTDEVICE_DECL constexpr explicit plus_equals(value_type v = value_type{}) noexcept : v_{std::move(v)} { }

  PETSC_NODISCARD PETSC_HOSTDEVICE_INLINE_DECL constexpr value_type operator()(const value_type &val) const noexcept { return val + v_; }

private:
  value_type v_;
};

template <typename T>
class times_equals {
public:
  using value_type = T;

  PETSC_HOSTDEVICE_DECL constexpr explicit times_equals(value_type v = value_type{}) noexcept : v_{std::move(v)} { }

  PETSC_NODISCARD PETSC_HOSTDEVICE_INLINE_DECL constexpr value_type operator()(const value_type &val) const noexcept { return val * v_; }

private:
  value_type v_;
};

template <typename T>
class axpy {
public:
  using value_type = T;

  PETSC_HOSTDEVICE_DECL constexpr explicit axpy(value_type v = value_type{}) noexcept : v_{std::move(v)} { }

  PETSC_NODISCARD PETSC_HOSTDEVICE_INLINE_DECL constexpr value_type operator()(const value_type &x, const value_type &y) const noexcept { return v_ * x + y; }

private:
  value_type v_;
};

namespace
{

template <typename T>
PETSC_HOSTDEVICE_INLINE_DECL constexpr plus_equals<T> make_plus_equals(const T &v) noexcept
{
  return plus_equals<T>{v};
}

template <typename T>
PETSC_HOSTDEVICE_INLINE_DECL constexpr times_equals<T> make_times_equals(const T &v) noexcept
{
  return times_equals<T>{v};
}

template <typename T>
PETSC_HOSTDEVICE_INLINE_DECL constexpr axpy<T> make_axpy(const T &v) noexcept
{
  return axpy<T>{v};
}

} // anonymous namespace

} // namespace functors

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_DEVICE_CUPM_KERNELS_HPP
