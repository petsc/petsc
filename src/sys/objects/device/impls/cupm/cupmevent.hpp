#ifndef PETSC_CUPMEVENT_HPP
#define PETSC_CUPMEVENT_HPP

#include <petsc/private/cupminterface.hpp>
#include <petsc/private/cpp/memory.hpp>
#include <petsc/private/cpp/object_pool.hpp>

#if defined(__cplusplus)
namespace Petsc
{

namespace device
{

namespace cupm
{

namespace
{

// A pool for allocating cupmEvent_t's. While events are generally very cheap to create and
// destroy, they are not free. Using the pool vs on-demand creation and destruction yields a ~20%
// speedup.
template <DeviceType T, unsigned long flags>
struct CUPMEventPoolAllocator : impl::Interface<T>, AllocatorBase<typename impl::Interface<T>::cupmEvent_t> {
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);

  PETSC_NODISCARD static PetscErrorCode create(cupmEvent_t *) noexcept;
  PETSC_NODISCARD static PetscErrorCode destroy(cupmEvent_t) noexcept;
};

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPoolAllocator<T, flags>::create(cupmEvent_t *event) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmEventCreateWithFlags(event, flags));
  PetscFunctionReturn(0);
}

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPoolAllocator<T, flags>::destroy(cupmEvent_t event) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmEventDestroy(event));
  PetscFunctionReturn(0);
}

} // anonymous namespace

template <DeviceType T, unsigned long flags, typename allocator_type = CUPMEventPoolAllocator<T, flags>, typename pool_type = ObjectPool<typename allocator_type::value_type, allocator_type>>
pool_type &cupm_event_pool() noexcept
{
  static pool_type pool;
  return pool;
}

// pool of events with timing disabled
template <DeviceType T>
inline auto cupm_fast_event_pool() noexcept -> decltype(cupm_event_pool<T, impl::Interface<T>::cupmEventDisableTiming>()) &
{
  return cupm_event_pool<T, impl::Interface<T>::cupmEventDisableTiming>();
}

// pool of events with timing enabled
template <DeviceType T>
inline auto cupm_timer_event_pool() noexcept -> decltype(cupm_event_pool<T, impl::Interface<T>::cupmEventDefault>()) &
{
  return cupm_event_pool<T, impl::Interface<T>::cupmEventDefault>();
}

// A simple wrapper of cupmEvent_t. This is used in conjunction with CUPMStream to build the
// event-stream pairing for the async allocator. It is also used as the data member of
// PetscEvent.
template <DeviceType T>
class CUPMEvent : impl::Interface<T>, public memory::PoolAllocated<CUPMEvent<T>> {
  using pool_type = memory::PoolAllocated<CUPMEvent<T>>;

public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type, T);

  constexpr CUPMEvent() noexcept = default;
  ~CUPMEvent() noexcept;

  CUPMEvent(CUPMEvent &&) noexcept;
  CUPMEvent &operator=(CUPMEvent &&) noexcept;

  // event is not copyable
  CUPMEvent(const CUPMEvent &)            = delete;
  CUPMEvent &operator=(const CUPMEvent &) = delete;

  PETSC_NODISCARD cupmEvent_t    get() noexcept;
  PETSC_NODISCARD PetscErrorCode record(cupmStream_t) noexcept;

  explicit operator bool() const noexcept;

private:
  cupmEvent_t event_{};
};

template <DeviceType T>
inline CUPMEvent<T>::~CUPMEvent() noexcept
{
  PetscFunctionBegin;
  if (event_) PetscCallAbort(PETSC_COMM_SELF, cupm_fast_event_pool<T>().deallocate(std::move(event_)));
  PetscFunctionReturnVoid();
}

template <DeviceType T>
inline CUPMEvent<T>::CUPMEvent(CUPMEvent &&other) noexcept : interface_type(std::move(other)), pool_type(std::move(other)), event_(util::exchange(other.event_, cupmEvent_t{}))
{
}

template <DeviceType T>
inline CUPMEvent<T> &CUPMEvent<T>::operator=(CUPMEvent &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    interface_type::operator=(std::move(other));
    pool_type::     operator=(std::move(other));
    if (event_) PetscCall(cupm_fast_event_pool<T>().deallocate(std::move(event_)));
    event_ = util::exchange(other.event_, cupmEvent_t{});
  }
  PetscFunctionReturn(*this);
}

template <DeviceType T>
inline typename CUPMEvent<T>::cupmEvent_t CUPMEvent<T>::get() noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!event_)) PetscCallAbort(PETSC_COMM_SELF, cupm_fast_event_pool<T>().allocate(&event_));
  PetscFunctionReturn(event_);
}

template <DeviceType T>
inline PetscErrorCode CUPMEvent<T>::record(cupmStream_t stream) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmEventRecord(get(), stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
inline CUPMEvent<T>::operator bool() const noexcept
{
  return event_ != cupmEvent_t{};
}

} // namespace cupm

} // namespace device

} // namespace Petsc
#endif // __cplusplus

#endif // PETSC_CUPMEVENT_HPP
