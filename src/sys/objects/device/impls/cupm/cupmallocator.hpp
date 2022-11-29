#ifndef CUPMALLOCATOR_HPP
#define CUPMALLOCATOR_HPP

#if defined(__cplusplus)
  #include <petsc/private/cpp/object_pool.hpp>

  #include "../segmentedmempool.hpp"
  #include "cupmthrustutility.hpp"

  #include <limits> // std::numeric_limits

namespace Petsc
{

namespace device
{

namespace cupm
{

// ==========================================================================================
// CUPM Host Allocator
// ==========================================================================================

template <DeviceType T, typename PetscType = char>
class HostAllocator;

// Allocator class to allocate pinned host memory for use with device
template <DeviceType T, typename PetscType>
class HostAllocator : public memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>, impl::Interface<T> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);
  using base_type = memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>;
  using typename base_type::real_value_type;
  using typename base_type::size_type;
  using typename base_type::value_type;

  template <typename U>
  PETSC_NODISCARD static PetscErrorCode allocate(value_type **, size_type, const StreamBase<U> *) noexcept;
  template <typename U>
  PETSC_NODISCARD static PetscErrorCode deallocate(value_type *, const StreamBase<U> *) noexcept;
  template <typename U>
  PETSC_NODISCARD static PetscErrorCode uninitialized_copy(value_type *, const value_type *, size_type, const StreamBase<U> *) noexcept;
};

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode HostAllocator<T, P>::allocate(value_type **ptr, size_type n, const StreamBase<U> *) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMMallocHost(ptr, n));
  PetscFunctionReturn(0);
}

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode HostAllocator<T, P>::deallocate(value_type *ptr, const StreamBase<U> *) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmFreeHost(ptr));
  PetscFunctionReturn(0);
}

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode HostAllocator<T, P>::uninitialized_copy(value_type *dest, const value_type *src, size_type n, const StreamBase<U> *stream) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMMemcpyAsync(dest, src, n, cupmMemcpyHostToHost, stream->get_stream(), true));
  PetscFunctionReturn(0);
}

// ==========================================================================================
// CUPM Device Allocator
// ==========================================================================================

template <DeviceType T, typename PetscType = char>
class DeviceAllocator;

template <DeviceType T, typename PetscType>
class DeviceAllocator : public memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>, impl::Interface<T> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);
  using base_type = memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>;
  using typename base_type::real_value_type;
  using typename base_type::size_type;
  using typename base_type::value_type;

  template <typename U>
  PETSC_NODISCARD static PetscErrorCode allocate(value_type **, size_type, const StreamBase<U> *) noexcept;
  template <typename U>
  PETSC_NODISCARD static PetscErrorCode deallocate(value_type *, const StreamBase<U> *) noexcept;
  template <typename U>
  PETSC_NODISCARD static PetscErrorCode zero(value_type *, size_type, const StreamBase<U> *) noexcept;
  template <typename U>
  PETSC_NODISCARD static PetscErrorCode uninitialized_copy(value_type *, const value_type *, size_type, const StreamBase<U> *) noexcept;
  template <typename U>
  PETSC_NODISCARD static PetscErrorCode set_canary(value_type *, size_type, const StreamBase<U> *) noexcept;
};

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode DeviceAllocator<T, P>::allocate(value_type **ptr, size_type n, const StreamBase<U> *stream) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMMallocAsync(ptr, n, stream->get_stream()));
  PetscFunctionReturn(0);
}

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode DeviceAllocator<T, P>::deallocate(value_type *ptr, const StreamBase<U> *stream) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmFreeAsync(ptr, stream->get_stream()));
  PetscFunctionReturn(0);
}

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode DeviceAllocator<T, P>::zero(value_type *ptr, size_type n, const StreamBase<U> *stream) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMMemsetAsync(ptr, 0, n, stream->get_stream(), true));
  PetscFunctionReturn(0);
}

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode DeviceAllocator<T, P>::uninitialized_copy(value_type *dest, const value_type *src, size_type n, const StreamBase<U> *stream) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMMemcpyAsync(dest, src, n, cupmMemcpyDeviceToDevice, stream->get_stream(), true));
  PetscFunctionReturn(0);
}

template <DeviceType T, typename P>
template <typename U>
inline PetscErrorCode DeviceAllocator<T, P>::set_canary(value_type *ptr, size_type n, const StreamBase<U> *stream) noexcept
{
  using limit_t           = std::numeric_limits<real_value_type>;
  const value_type canary = limit_t::has_signaling_NaN ? limit_t::signaling_NaN() : limit_t::max();

  PetscFunctionBegin;
  PetscCall(impl::ThrustSet<T>(stream->get_stream(), n, ptr, &canary));
  PetscFunctionReturn(0);
}

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // CUPMALLOCATOR_HPP
