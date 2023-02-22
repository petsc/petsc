#ifndef PETSC_CUPMEVENT_HPP
#define PETSC_CUPMEVENT_HPP

#include <petsc/private/cupminterface.hpp>
#include <petsc/private/cpp/memory.hpp>
#include <petsc/private/cpp/object_pool.hpp>

#if defined(__cplusplus)
  #include <stack>
namespace Petsc
{

namespace device
{

namespace cupm
{

// A pool for allocating cupmEvent_t's. While events are generally very cheap to create and
// destroy, they are not free. Using the pool vs on-demand creation and destruction yields a ~20%
// speedup.
template <DeviceType T, unsigned long flags>
class CUPMEventPool : impl::Interface<T>, public RegisterFinalizeable<CUPMEventPool<T, flags>> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  PetscErrorCode allocate(cupmEvent_t *) noexcept;
  PetscErrorCode deallocate(cupmEvent_t *) noexcept;

  PetscErrorCode finalize_() noexcept;

private:
  std::stack<cupmEvent_t> pool_;
};

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPool<T, flags>::finalize_() noexcept
{
  PetscFunctionBegin;
  while (!pool_.empty()) {
    PetscCallCUPM(cupmEventDestroy(std::move(pool_.top())));
    PetscCallCXX(pool_.pop());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPool<T, flags>::allocate(cupmEvent_t *event) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(event, 1);
  if (pool_.empty()) {
    PetscCall(this->register_finalize());
    PetscCallCUPM(cupmEventCreateWithFlags(event, flags));
  } else {
    PetscCallCXX(*event = std::move(pool_.top()));
    PetscCallCXX(pool_.pop());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPool<T, flags>::deallocate(cupmEvent_t *in_event) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(in_event, 1);
  if (auto event = std::exchange(*in_event, cupmEvent_t{})) {
    if (this->registered()) {
      PetscCallCXX(pool_.push(std::move(event)));
    } else {
      PetscCallCUPM(cupmEventDestroy(event));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, unsigned long flags>
CUPMEventPool<T, flags> &cupm_event_pool() noexcept
{
  static CUPMEventPool<T, flags> pool;
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
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  constexpr CUPMEvent() noexcept = default;
  ~CUPMEvent() noexcept;

  CUPMEvent(CUPMEvent &&) noexcept;
  CUPMEvent &operator=(CUPMEvent &&) noexcept;

  // event is not copyable
  CUPMEvent(const CUPMEvent &)            = delete;
  CUPMEvent &operator=(const CUPMEvent &) = delete;

  PETSC_NODISCARD cupmEvent_t get() noexcept;
  PetscErrorCode              record(cupmStream_t) noexcept;

  explicit operator bool() const noexcept;

private:
  cupmEvent_t event_{};
};

template <DeviceType T>
inline CUPMEvent<T>::~CUPMEvent() noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, cupm_fast_event_pool<T>().deallocate(&event_));
  PetscFunctionReturnVoid();
}

template <DeviceType T>
inline CUPMEvent<T>::CUPMEvent(CUPMEvent &&other) noexcept : pool_type(std::move(other)), event_(util::exchange(other.event_, cupmEvent_t{}))
{
  static_assert(std::is_empty<impl::Interface<T>>::value, "");
}

template <DeviceType T>
inline CUPMEvent<T> &CUPMEvent<T>::operator=(CUPMEvent &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    pool_type::operator=(std::move(other));
    PetscCallAbort(PETSC_COMM_SELF, cupm_fast_event_pool<T>().deallocate(&event_));
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
