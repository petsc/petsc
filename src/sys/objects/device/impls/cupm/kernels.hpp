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

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_DEVICE_CUPM_KERNELS_HPP
