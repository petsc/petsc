#ifndef PETSC_CUPMSTREAM_HPP
#define PETSC_CUPMSTREAM_HPP

#include <petsc/private/cupminterface.hpp>

#include "../segmentedmempool.hpp"
#include "cupmevent.hpp"

#if defined(__cplusplus)
namespace Petsc
{

namespace device
{

namespace cupm
{

// A bare wrapper around a cupmStream_t. The reason it exists is because we need to uniquely
// identify separate cupm streams. This is so that the memory pool can accelerate allocation
// calls as it can just pass back a pointer to memory that was used on the same
// stream. Otherwise it must either serialize with another stream or allocate a new chunk.
// Address of the objects does not suffice since cupmStreams are very likely internally reused.

template <DeviceType T>
class CUPMStream : public StreamBase<CUPMStream<T>>, impl::Interface<T> {
  using crtp_base_type = StreamBase<CUPMStream<T>>;
  friend crtp_base_type;

public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  using stream_type = cupmStream_t;
  using id_type     = typename crtp_base_type::id_type;
  using event_type  = CUPMEvent<T>;
  using flag_type   = unsigned int;

  CUPMStream() noexcept = default;

  PetscErrorCode destroy() noexcept;
  PetscErrorCode create(flag_type) noexcept;
  PetscErrorCode change_type(PetscStreamType) noexcept;

private:
  stream_type stream_{};
  id_type     id_ = new_id_();

  PETSC_NODISCARD static id_type new_id_() noexcept;

  // CRTP implementations
  PETSC_NODISCARD stream_type get_stream_() const noexcept;
  PETSC_NODISCARD id_type     get_id_() const noexcept;
  PetscErrorCode              record_event_(event_type &) const noexcept;
  PetscErrorCode              wait_for_(event_type &) const noexcept;
};

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::destroy() noexcept
{
  PetscFunctionBegin;
  if (stream_) {
    PetscCallCUPM(cupmStreamDestroy(stream_));
    stream_ = cupmStream_t{};
    id_     = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::create(flag_type flags) noexcept
{
  PetscFunctionBegin;
  if (stream_) {
    if (PetscDefined(USE_DEBUG)) {
      flag_type current_flags;

      PetscCallCUPM(cupmStreamGetFlags(stream_, &current_flags));
      PetscCheck(flags == current_flags, PETSC_COMM_SELF, PETSC_ERR_GPU, "Current flags %u != requested flags %u for stream %d", current_flags, flags, id_);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCallCUPM(cupmStreamCreateWithFlags(&stream_, flags));
  id_ = new_id_();
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::change_type(PetscStreamType newtype) noexcept
{
  PetscFunctionBegin;
  if (newtype == PETSC_STREAM_GLOBAL_BLOCKING) {
    PetscCall(destroy());
  } else {
    const flag_type preferred = newtype == PETSC_STREAM_DEFAULT_BLOCKING ? cupmStreamDefault : cupmStreamNonBlocking;

    if (stream_) {
      flag_type flag;

      PetscCallCUPM(cupmStreamGetFlags(stream_, &flag));
      if (flag == preferred) PetscFunctionReturn(PETSC_SUCCESS);
      PetscCall(destroy());
    }
    PetscCall(create(preferred));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline typename CUPMStream<T>::id_type CUPMStream<T>::new_id_() noexcept
{
  static id_type id = 0;
  return id++;
}

// CRTP implementations
template <DeviceType T>
inline typename CUPMStream<T>::stream_type CUPMStream<T>::get_stream_() const noexcept
{
  return stream_;
}

template <DeviceType T>
inline typename CUPMStream<T>::id_type CUPMStream<T>::get_id_() const noexcept
{
  return id_;
}

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::record_event_(event_type &event) const noexcept
{
  PetscFunctionBegin;
  PetscCall(event.record(stream_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::wait_for_(event_type &event) const noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmStreamWaitEvent(stream_, event.get(), 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace cupm

} // namespace device

} // namespace Petsc
#endif // __cplusplus

#endif // PETSC_CUPMSTREAM_HPP
