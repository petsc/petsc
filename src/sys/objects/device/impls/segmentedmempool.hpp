#ifndef PETSC_SEGMENTEDMEMPOOL_HPP
#define PETSC_SEGMENTEDMEMPOOL_HPP

#include <petsc/private/deviceimpl.h>

#include <petsc/private/cpp/macros.hpp>
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/memory.hpp>

#include <limits>
#include <deque>
#include <vector>

namespace Petsc
{

namespace device
{

template <typename T>
class StreamBase {
public:
  using id_type      = int;
  using derived_type = T;

  static const id_type INVALID_ID;

  // needed so that dependent auto works, see veccupmimpl.h for a detailed discussion
  template <typename U = T>
  PETSC_NODISCARD auto get_stream() const noexcept PETSC_DECLTYPE_AUTO_RETURNS(static_cast<const U &>(*this).get_stream_());

  PETSC_NODISCARD id_type get_id() const noexcept { return static_cast<const T &>(*this).get_id_(); }

  template <typename E>
  PetscErrorCode record_event(E &&event) const noexcept
  {
    return static_cast<const T &>(*this).record_event_(std::forward<E>(event));
  }

  template <typename E>
  PetscErrorCode wait_for_event(E &&event) const noexcept
  {
    return static_cast<const T &>(*this).wait_for_(std::forward<E>(event));
  }

protected:
  constexpr StreamBase() noexcept = default;

  struct default_event_type { };
  using default_stream_type = std::nullptr_t;

  PETSC_NODISCARD static constexpr default_stream_type get_stream_() noexcept { return nullptr; }

  PETSC_NODISCARD static constexpr id_type get_id_() noexcept { return 0; }

  template <typename U = T>
  static constexpr PetscErrorCode record_event_(const typename U::event_type &) noexcept
  {
    return PETSC_SUCCESS;
  }

  template <typename U = T>
  static constexpr PetscErrorCode wait_for_(const typename U::event_type &) noexcept
  {
    return PETSC_SUCCESS;
  }
};

template <typename T>
const typename StreamBase<T>::id_type StreamBase<T>::INVALID_ID = -1;

struct DefaultStream : StreamBase<DefaultStream> {
  using stream_type = typename StreamBase<DefaultStream>::default_stream_type;
  using id_type     = typename StreamBase<DefaultStream>::id_type;
  using event_type  = typename StreamBase<DefaultStream>::default_event_type;
};

} // namespace device

namespace memory
{

namespace impl
{

// ==========================================================================================
// MemoryChunk
//
// Represents a checked-out region of a MemoryBlock. Tracks the offset into the owning
// MemoryBlock and its size/capacity
// ==========================================================================================

template <typename EventType>
class MemoryChunk {
public:
  using event_type = EventType;
  using size_type  = std::size_t;

  MemoryChunk(size_type, size_type) noexcept;
  explicit MemoryChunk(size_type) noexcept;

  MemoryChunk(MemoryChunk &&) noexcept;
  MemoryChunk &operator=(MemoryChunk &&) noexcept;

  MemoryChunk(const MemoryChunk &) noexcept            = delete;
  MemoryChunk &operator=(const MemoryChunk &) noexcept = delete;

  PETSC_NODISCARD size_type start() const noexcept { return start_; }
  PETSC_NODISCARD size_type size() const noexcept { return size_; }
  // REVIEW ME:
  // make this an actual field, normally each chunk shrinks_to_fit() on begin claimed, but in
  // theory only the last chunk needs to do this
  PETSC_NODISCARD size_type capacity() const noexcept { return size_; }
  PETSC_NODISCARD size_type total_offset() const noexcept { return start() + size(); }

  template <typename U>
  PetscErrorCode release(const device::StreamBase<U> *) noexcept;
  template <typename U>
  PetscErrorCode claim(const device::StreamBase<U> *, size_type, bool *, bool = false) noexcept;
  template <typename U>
  PETSC_NODISCARD bool can_claim(const device::StreamBase<U> *, size_type, bool) const noexcept;
  PetscErrorCode       resize(size_type) noexcept;
  PETSC_NODISCARD bool contains(size_type) const noexcept;

private:
  // clang-format off
  event_type      event_{};          // event recorded when the chunk was released
  bool            open_      = true; // is this chunk open?
  // id of the last stream to use the chunk, populated on release
  int             stream_id_ = device::DefaultStream::INVALID_ID;
  size_type       size_      = 0;    // size of the chunk
  const size_type start_     = 0;    // offset from the start of the owning block
  // clang-format on

  template <typename U>
  PETSC_NODISCARD bool stream_compat_(const device::StreamBase<U> *) const noexcept;
};

// ==========================================================================================
// MemoryChunk - Private API
// ==========================================================================================

// asks and answers the question: can this stream claim this chunk without serializing?
template <typename E>
template <typename U>
inline bool MemoryChunk<E>::stream_compat_(const device::StreamBase<U> *strm) const noexcept
{
  return (stream_id_ == strm->INVALID_ID) || (stream_id_ == strm->get_id());
}

// ==========================================================================================
// MemoryChunk - Public API
// ==========================================================================================

template <typename E>
inline MemoryChunk<E>::MemoryChunk(size_type start, size_type size) noexcept : size_(size), start_(start)
{
}

template <typename E>
inline MemoryChunk<E>::MemoryChunk(size_type size) noexcept : MemoryChunk(0, size)
{
}

template <typename E>
inline MemoryChunk<E>::MemoryChunk(MemoryChunk<E> &&other) noexcept :
  event_(std::move(other.event_)), open_(util::exchange(other.open_, false)), stream_id_(util::exchange(other.stream_id_, device::DefaultStream::INVALID_ID)), size_(util::exchange(other.size_, 0)), start_(std::move(other.start_))
{
}

template <typename E>
inline MemoryChunk<E> &MemoryChunk<E>::operator=(MemoryChunk<E> &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    event_     = std::move(other.event_);
    open_      = util::exchange(other.open_, false);
    stream_id_ = util::exchange(other.stream_id_, device::DefaultStream::INVALID_ID);
    size_      = util::exchange(other.size_, 0);
    start_     = std::move(other.start_);
  }
  PetscFunctionReturn(*this);
}

/*
  MemoryChunk::release - release a chunk on a stream

  Input Parameter:
. stream - the stream to release the chunk with

  Notes:
  Inserts a release operation on stream and records the state of stream at the time this
  routine was called.

  Future allocation requests which attempt to claim the chunk on the same stream may re-acquire
  the chunk without serialization.

  If another stream attempts to claim the chunk they must wait for the recorded event before
  claiming the chunk.
*/
template <typename E>
template <typename U>
inline PetscErrorCode MemoryChunk<E>::release(const device::StreamBase<U> *stream) noexcept
{
  PetscFunctionBegin;
  open_      = true;
  stream_id_ = stream->get_id();
  PetscCall(stream->record_event(event_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryChunk::claim - attempt to claim a particular chunk

  Input Parameters:
+ stream    - the stream on which to attempt to claim
. req_size  - the requested size (in elements) to attempt to claim
- serialize - (optional, false) whether the claimant allows serialization

  Output Parameter:
. success - true if the chunk was claimed, false otherwise
*/
template <typename E>
template <typename U>
inline PetscErrorCode MemoryChunk<E>::claim(const device::StreamBase<U> *stream, size_type req_size, bool *success, bool serialize) noexcept
{
  PetscFunctionBegin;
  if ((*success = can_claim(stream, req_size, serialize))) {
    if (serialize && !stream_compat_(stream)) PetscCall(stream->wait_for_event(event_));
    PetscCall(resize(req_size));
    open_ = false;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryChunk::can_claim - test whether a particular chunk can be claimed

  Input Parameters:
+ stream    - the stream on which to attempt to claim
. req_size  - the requested size (in elements) to attempt to claim
- serialize - whether the claimant allows serialization

  Output:
. [return] - true if the chunk is claimable given the configuration, false otherwise
*/
template <typename E>
template <typename U>
inline bool MemoryChunk<E>::can_claim(const device::StreamBase<U> *stream, size_type req_size, bool serialize) const noexcept
{
  if (open_ && (req_size <= capacity())) {
    // fully compatible
    if (stream_compat_(stream)) return true;
    // stream wasn't compatible, but could claim if we serialized
    if (serialize) return true;
    // incompatible stream and did not want to serialize
  }
  return false;
}

/*
  MemoryChunk::resize - grow a chunk to new size

  Input Parameter:
. newsize - the new size Requested

  Notes:
  newsize cannot be larger than capacity
*/
template <typename E>
inline PetscErrorCode MemoryChunk<E>::resize(size_type newsize) noexcept
{
  PetscFunctionBegin;
  PetscAssert(newsize <= capacity(), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "New size %zu larger than capacity %zu", newsize, capacity());
  size_ = newsize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryChunk::contains - query whether a memory chunk contains a particular offset

  Input Parameters:
. offset - The offset from the MemoryBlock start

  Notes:
  Returns true if the chunk contains the offset, false otherwise
*/
template <typename E>
inline bool MemoryChunk<E>::contains(size_type offset) const noexcept
{
  return (offset >= start()) && (offset < total_offset());
}

// ==========================================================================================
// MemoryBlock
//
// A "memory block" manager, which owns the pointer to a particular memory range. Retrieving
// and restoring a block is thread-safe (so may be used by multiple device streams).
// ==========================================================================================

template <typename T, typename AllocatorType, typename StreamType>
class MemoryBlock {
public:
  using value_type      = T;
  using allocator_type  = AllocatorType;
  using stream_type     = StreamType;
  using event_type      = typename stream_type::event_type;
  using chunk_type      = MemoryChunk<event_type>;
  using size_type       = typename chunk_type::size_type;
  using chunk_list_type = std::vector<chunk_type>;

  template <typename U>
  MemoryBlock(allocator_type *, size_type, const device::StreamBase<U> *) noexcept;

  ~MemoryBlock() noexcept(std::is_nothrow_destructible<chunk_list_type>::value);

  MemoryBlock(MemoryBlock &&) noexcept;
  MemoryBlock &operator=(MemoryBlock &&) noexcept;

  // memory blocks are not copyable
  MemoryBlock(const MemoryBlock &)            = delete;
  MemoryBlock &operator=(const MemoryBlock &) = delete;

  /* --- actual functions --- */
  PetscErrorCode       try_allocate_chunk(size_type, T **, const stream_type *, bool *) noexcept;
  PetscErrorCode       try_deallocate_chunk(T **, const stream_type *, bool *) noexcept;
  PetscErrorCode       try_find_chunk(const T *, chunk_type **) noexcept;
  PETSC_NODISCARD bool owns_pointer(const T *) const noexcept;

  PETSC_NODISCARD size_type size() const noexcept { return size_; }
  PETSC_NODISCARD size_type bytes() const noexcept { return sizeof(value_type) * size(); }
  PETSC_NODISCARD size_type num_chunks() const noexcept { return chunks_.size(); }

private:
  value_type     *mem_{};
  allocator_type *allocator_{};
  size_type       size_{};
  chunk_list_type chunks_{};

  PetscErrorCode clear_(const stream_type *) noexcept;
};

// ==========================================================================================
// MemoryBlock - Private API
// ==========================================================================================

// clear the memory block, called from destructors and move assignment/construction
template <typename T, typename A, typename S>
PetscErrorCode MemoryBlock<T, A, S>::clear_(const stream_type *stream) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(mem_)) {
    PetscCall(allocator_->deallocate(mem_, stream));
    mem_ = nullptr;
  }
  size_ = 0;
  PetscCallCXX(chunks_.clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MemoryBlock - Public API
// ==========================================================================================

// default constructor, allocates memory immediately
template <typename T, typename A, typename S>
template <typename U>
MemoryBlock<T, A, S>::MemoryBlock(allocator_type *alloc, size_type s, const device::StreamBase<U> *stream) noexcept : allocator_(alloc), size_(s)
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, alloc->allocate(&mem_, s, stream));
  PetscAssertAbort(mem_, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory block of size %zu", s);
  PetscFunctionReturnVoid();
}

template <typename T, typename A, typename S>
MemoryBlock<T, A, S>::~MemoryBlock() noexcept(std::is_nothrow_destructible<chunk_list_type>::value)
{
  stream_type stream;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, clear_(&stream));
  PetscFunctionReturnVoid();
}

template <typename T, typename A, typename S>
MemoryBlock<T, A, S>::MemoryBlock(MemoryBlock &&other) noexcept : mem_(util::exchange(other.mem_, nullptr)), allocator_(other.allocator_), size_(util::exchange(other.size_, 0)), chunks_(std::move(other.chunks_))
{
}

template <typename T, typename A, typename S>
MemoryBlock<T, A, S> &MemoryBlock<T, A, S>::operator=(MemoryBlock &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    stream_type stream;

    PetscCallAbort(PETSC_COMM_SELF, clear_(&stream));
    mem_       = util::exchange(other.mem_, nullptr);
    allocator_ = other.allocator_;
    size_      = util::exchange(other.size_, 0);
    chunks_    = std::move(other.chunks_);
  }
  PetscFunctionReturn(*this);
}

/*
  MemoryBock::owns_pointer - returns true if this block owns a pointer, false otherwise
*/
template <typename T, typename A, typename S>
inline bool MemoryBlock<T, A, S>::owns_pointer(const T *ptr) const noexcept
{
  // each pool is linear in memory, so it suffices to check the bounds
  return (ptr >= mem_) && (ptr < std::next(mem_, size()));
}

/*
  MemoryBlock::try_allocate_chunk - try to get a chunk from this MemoryBlock

  Input Parameters:
+ req_size - the requested size of the allocation (in elements)
. ptr      - ptr to fill
- stream   - stream to fill the pointer on

  Output Parameter:
. success  - true if chunk was gotten, false otherwise

  Notes:
  If the current memory could not satisfy the memory request, ptr is unchanged
*/
template <typename T, typename A, typename S>
inline PetscErrorCode MemoryBlock<T, A, S>::try_allocate_chunk(size_type req_size, T **ptr, const stream_type *stream, bool *success) noexcept
{
  PetscFunctionBegin;
  *success = false;
  if (req_size <= size()) {
    const auto try_create_chunk = [&]() {
      const auto was_empty     = chunks_.empty();
      const auto block_alloced = was_empty ? 0 : chunks_.back().total_offset();

      PetscFunctionBegin;
      if (block_alloced + req_size <= size()) {
        PetscCallCXX(chunks_.emplace_back(block_alloced, req_size));
        PetscCall(chunks_.back().claim(stream, req_size, success));
        *ptr = mem_ + block_alloced;
        if (was_empty) PetscAssert(*success, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to claim chunk (of size %zu) even though block (of size %zu) was empty!", req_size, size());
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto try_find_open_chunk = [&](bool serialize = false) {
      PetscFunctionBegin;
      for (auto &chunk : chunks_) {
        PetscCall(chunk.claim(stream, req_size, success, serialize));
        if (*success) {
          *ptr = mem_ + chunk.start();
          break;
        }
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto try_steal_other_stream_chunk = [&]() {
      PetscFunctionBegin;
      PetscCall(try_find_open_chunk(true));
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    // search previously distributed chunks, but only claim one if it is on the same stream
    // as us
    PetscCall(try_find_open_chunk());

    // if we are here we couldn't reuse one of our own chunks so check first if the pool
    // has room for a new one
    if (!*success) PetscCall(try_create_chunk());

    // try pruning dead chunks off the back, note we do this regardless of whether we are
    // successful
    while (chunks_.back().can_claim(stream, 0, false)) {
      PetscCallCXX(chunks_.pop_back());
      if (chunks_.empty()) {
        // if chunks are empty it implies we have managed to claim (and subsequently destroy)
        // our own chunk twice! something has gone wrong
        PetscAssert(!*success, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Successfully claimed a chunk (of size %zu, from block of size %zu) but have now managed to claim it for a second time (and destroyed it)!", req_size, size());
        break;
      }
    }

    // if previously unsuccessful see if enough space has opened up due to pruning. note that
    // if the chunk list was emptied from the pruning this call must succeed in allocating a
    // chunk, otherwise something is wrong
    if (!*success) PetscCall(try_create_chunk());

    // last resort, iterate over all chunks and see if we can steal one by waiting on the
    // current owner to finish using it
    if (!*success) PetscCall(try_steal_other_stream_chunk());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryBlock::try_deallocate_chunk - try to restore a chunk to this MemoryBlock

  Input Parameters:
+ ptr     - ptr to restore
- stream  - stream to restore the pointer on

  Output Parameter:
. success - true if chunk was restored, false otherwise

  Notes:
  ptr is set to nullptr on successful restore, and is unchanged otherwise. If the ptr is owned
  by this MemoryBlock then it is restored on stream. The same stream may receive ptr again
  without synchronization, but other streams may not do so until either serializing or the
  stream is idle again.
*/
template <typename T, typename A, typename S>
inline PetscErrorCode MemoryBlock<T, A, S>::try_deallocate_chunk(T **ptr, const stream_type *stream, bool *success) noexcept
{
  chunk_type *chunk = nullptr;

  PetscFunctionBegin;
  PetscCall(try_find_chunk(*ptr, &chunk));
  if (chunk) {
    PetscCall(chunk->release(stream));
    *ptr     = nullptr;
    *success = true;
  } else {
    *success = false;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryBlock::try_find_chunk - try to find the chunk which owns ptr

  Input Parameter:
. ptr - the pointer to look for

  Output Parameter:
. ret_chunk - pointer to the owning chunk or nullptr if not found
*/
template <typename T, typename A, typename S>
inline PetscErrorCode MemoryBlock<T, A, S>::try_find_chunk(const T *ptr, chunk_type **ret_chunk) noexcept
{
  PetscFunctionBegin;
  *ret_chunk = nullptr;
  if (owns_pointer(ptr)) {
    const auto offset = static_cast<size_type>(ptr - mem_);

    for (auto &chunk : chunks_) {
      if (chunk.contains(offset)) {
        *ret_chunk = &chunk;
        break;
      }
    }

    PetscAssert(*ret_chunk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to find %zu in block, even though it is within block range [%zu, %zu)", reinterpret_cast<uintptr_t>(ptr), reinterpret_cast<uintptr_t>(mem_), reinterpret_cast<uintptr_t>(std::next(mem_, size())));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace detail
{

template <typename T>
struct real_type {
  using type = T;
};

template <>
struct real_type<PetscScalar> {
  using type = PetscReal;
};

} // namespace detail

template <typename T>
struct SegmentedMemoryPoolAllocatorBase {
  using value_type      = T;
  using size_type       = std::size_t;
  using real_value_type = typename detail::real_type<T>::type;

  template <typename U>
  static PetscErrorCode allocate(value_type **, size_type, const device::StreamBase<U> *) noexcept;
  template <typename U>
  static PetscErrorCode deallocate(value_type *, const device::StreamBase<U> *) noexcept;
  template <typename U>
  static PetscErrorCode zero(value_type *, size_type, const device::StreamBase<U> *) noexcept;
  template <typename U>
  static PetscErrorCode uninitialized_copy(value_type *, const value_type *, size_type, const device::StreamBase<U> *) noexcept;
  template <typename U>
  static PetscErrorCode set_canary(value_type *, size_type, const device::StreamBase<U> *) noexcept;
};

template <typename T>
template <typename U>
inline PetscErrorCode SegmentedMemoryPoolAllocatorBase<T>::allocate(value_type **ptr, size_type n, const device::StreamBase<U> *) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscMalloc1(n, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename U>
inline PetscErrorCode SegmentedMemoryPoolAllocatorBase<T>::deallocate(value_type *ptr, const device::StreamBase<U> *) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscFree(ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename U>
inline PetscErrorCode SegmentedMemoryPoolAllocatorBase<T>::zero(value_type *ptr, size_type n, const device::StreamBase<U> *) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscArrayzero(ptr, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename U>
inline PetscErrorCode SegmentedMemoryPoolAllocatorBase<T>::uninitialized_copy(value_type *dest, const value_type *src, size_type n, const device::StreamBase<U> *) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscArraycpy(dest, src, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename U>
inline PetscErrorCode SegmentedMemoryPoolAllocatorBase<T>::set_canary(value_type *ptr, size_type n, const device::StreamBase<U> *) noexcept
{
  using limit_type            = std::numeric_limits<real_value_type>;
  constexpr value_type canary = limit_type::has_signaling_NaN ? limit_type::signaling_NaN() : limit_type::max();

  PetscFunctionBegin;
  for (size_type i = 0; i < n; ++i) ptr[i] = canary;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

// ==========================================================================================
// SegmentedMemoryPool
//
// Stream-aware async memory allocator. Holds a list of memory "blocks" which each control an
// allocated buffer. This buffer is further split into memory "chunks" which control
// consecutive, non-overlapping regions of the block. Chunks may be in 1 of 2 states:
//
// 1. Open:
//    The chunk is free to be claimed by the next suitable allocation request. If the
//    allocation request is made on the same stream as the chunk was deallocated on, no
//    serialization needs to occur. If not, the allocating stream must wait for the
//    event. Claiming the chunk "closes" the chunk.
//
// 2. Closed:
//    The chunk has been claimed by an allocation request. It cannot be opened again until it
//    is deallocated; doing so "opens" the chunk.
//
// Note that there does not need to be a chunk for every region, chunks are created to satisfy
// an allocation request.
//
// Thus there is usually a region of "unallocated" memory at the end of the buffer, which may
// be claimed by a newly created chunk if existing chunks cannot satisfy the allocation
// request. This region exists _only_ at the end, as there are no gaps between chunks.
//
//
// |-----------------------------------------------------------------------------------------
// | SegmentedMemoryPool
// |
// | ||-------------||
// | ||             ||    -------------------------------------------------------------------
// | ||             ||    | AAAAAAAAAAAAAABBBBBBBCCCCCCCCCCCCCCCCCCCCDDDDDDDDDDDDDXXXXXXXX...
// | ||             ||    | |             |      |                   |            |
// | ||             ||    | x-----x-------x-----xx---------x---------x------x-----x
// | || MemoryBlock || -> | ------|-------------|----------|----------------|--------
// | ||             ||    | | MemoryChunk | MemoryChunk | MemoryChunk | MemoryChunk |
// | ||             ||    | ---------------------------------------------------------
// | ||             ||    -------------------------------------------------------------------
// | ||-------------||
// | ||             ||
// | ||     ...     ||
// | ||             ||
// ==========================================================================================

template <typename MemType, typename StreamType = device::DefaultStream, typename AllocType = impl::SegmentedMemoryPoolAllocatorBase<MemType>, std::size_t DefaultChunkSize = 256>
class SegmentedMemoryPool;

// The actual memory pool class. It is in essence just a wrapper for a list of MemoryBlocks.
template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
class SegmentedMemoryPool : public RegisterFinalizeable<SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>> {
public:
  using value_type     = MemType;
  using stream_type    = StreamType;
  using allocator_type = AllocType;
  using block_type     = impl::MemoryBlock<value_type, allocator_type, stream_type>;
  using pool_type      = std::deque<block_type>;
  using size_type      = typename block_type::size_type;

  explicit SegmentedMemoryPool(AllocType = AllocType{}, std::size_t = DefaultChunkSize) noexcept(std::is_nothrow_default_constructible<pool_type>::value);

  PetscErrorCode allocate(PetscInt, value_type **, const stream_type *, size_type = std::alignment_of<MemType>::value) noexcept;
  PetscErrorCode deallocate(value_type **, const stream_type *) noexcept;
  PetscErrorCode reallocate(PetscInt, value_type **, const stream_type *) noexcept;

private:
  pool_type      pool_;
  allocator_type allocator_;
  size_type      chunk_size_;

  PetscErrorCode make_block_(size_type, const stream_type *) noexcept;

  friend class RegisterFinalizeable<SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>>;
  PetscErrorCode register_finalize_(const stream_type *) noexcept;
  PetscErrorCode finalize_() noexcept;

  PetscErrorCode allocate_(size_type, value_type **, const stream_type *) noexcept;
};

// ==========================================================================================
// SegmentedMemoryPool - Private API
// ==========================================================================================

template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::make_block_(size_type size, const stream_type *stream) noexcept
{
  const auto block_size = std::max(size, chunk_size_);

  PetscFunctionBegin;
  PetscCallCXX(pool_.emplace_back(&allocator_, block_size, stream));
  PetscCall(PetscInfo(nullptr, "Allocated new block of size %zu, total %zu blocks\n", block_size, pool_.size()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::register_finalize_(const stream_type *stream) noexcept
{
  PetscFunctionBegin;
  PetscCall(make_block_(chunk_size_, stream));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::finalize_() noexcept
{
  PetscFunctionBegin;
  PetscCallCXX(pool_.clear());
  chunk_size_ = DefaultChunkSize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::allocate_(size_type size, value_type **ptr, const stream_type *stream) noexcept
{
  auto found = false;

  PetscFunctionBegin;
  PetscCall(this->register_finalize(stream));
  for (auto &block : pool_) {
    PetscCall(block.try_allocate_chunk(size, ptr, stream, &found));
    if (PetscLikely(found)) PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscInfo(nullptr, "Could not find an open block in the pool (%zu blocks) (requested size %zu), allocating new block\n", pool_.size(), size));
  // if we are here we couldn't find an open block in the pool, so make a new block
  PetscCall(make_block_(size, stream));
  // and assign it
  PetscCall(pool_.back().try_allocate_chunk(size, ptr, stream, &found));
  PetscAssert(found, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to get a suitable memory chunk (of size %zu) from newly allocated memory block (size %zu)", size, pool_.back().size());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// SegmentedMemoryPool - Public API
// ==========================================================================================

template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::SegmentedMemoryPool(AllocType alloc, std::size_t size) noexcept(std::is_nothrow_default_constructible<pool_type>::value) : allocator_(std::move(alloc)), chunk_size_(size)
{
}

/*
  SegmentedMemoryPool::allocate - get an allocation from the memory pool

  Input Parameters:
+ req_size - size (in elements) to get
. ptr      - the pointer to hold the allocation
- stream   - the stream on which to get the allocation

  Output Parameter:
. ptr - the pointer holding the allocation

  Notes:
  req_size cannot be negative. If req_size if zero, ptr is set to nullptr
*/
template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::allocate(PetscInt req_size, value_type **ptr, const stream_type *stream, size_type alignment) noexcept
{
  value_type *ret_ptr = nullptr;

  PetscFunctionBegin;
  PetscAssert(req_size >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested memory amount (%" PetscInt_FMT ") must be >= 0", req_size);
  PetscValidPointer(ptr, 2);
  PetscValidPointer(stream, 3);
  if (req_size) {
    const auto size         = static_cast<size_type>(req_size);
    auto       aligned_size = alignment == alignof(char) ? size : size + alignment;
    void      *vptr         = nullptr;

    PetscCall(allocate_(aligned_size, &ret_ptr, stream));
    vptr = ret_ptr;
    std::align(alignment, size, vptr, aligned_size);
    ret_ptr = reinterpret_cast<value_type *>(vptr);
    // sets memory to NaN or infinity depending on the type to catch out uninitialized memory
    // accesses.
    if (PetscDefined(USE_DEBUG)) PetscCall(allocator_.set_canary(ret_ptr, size, stream));
  }
  *ptr = ret_ptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SegmentedMemoryPool::deallocate - release a pointer back to the memory pool

  Input Parameters:
+ ptr    - the pointer to release
- stream - the stream to release it on

  Notes:
  If ptr is not owned by the pool it is unchanged.
*/
template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::deallocate(value_type **ptr, const stream_type *stream) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(ptr, 1);
  PetscValidPointer(stream, 2);
  // nobody owns a nullptr, and if they do then they have bigger problems
  if (!*ptr) PetscFunctionReturn(PETSC_SUCCESS);
  for (auto &block : pool_) {
    auto found = false;

    PetscCall(block.try_deallocate_chunk(ptr, stream, &found));
    if (PetscLikely(found)) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SegmentedMemoryPool::reallocate - Resize an allocated buffer

  Input Parameters:
+ new_req_size - the new buffer size
. ptr          - pointer to the buffer
- stream       - stream to resize with

  Output Parameter:
. ptr - pointer to the new region

  Notes:
  ptr must have been allocated by the pool.

  It's OK to shrink the buffer, even down to 0 (in which case it is just deallocated).
*/
template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::reallocate(PetscInt new_req_size, value_type **ptr, const stream_type *stream) noexcept
{
  using chunk_type = typename block_type::chunk_type;

  const auto  new_size = static_cast<size_type>(new_req_size);
  const auto  old_ptr  = *ptr;
  chunk_type *chunk    = nullptr;

  PetscFunctionBegin;
  PetscAssert(new_req_size >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested memory amount (%" PetscInt_FMT ") must be >= 0", new_req_size);
  PetscValidPointer(ptr, 2);
  PetscValidPointer(stream, 3);

  // if reallocating to zero, just free
  if (PetscUnlikely(new_size == 0)) {
    PetscCall(deallocate(ptr, stream));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // search the blocks for the owning chunk
  for (auto &block : pool_) {
    PetscCall(block.try_find_chunk(old_ptr, &chunk));
    if (chunk) break; // found
  }
  PetscAssert(chunk, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Memory pool does not own %p, so cannot reallocate it", *ptr);

  if (chunk->capacity() < new_size) {
    // chunk does not have enough room, need to grab a fresh chunk and copy to it
    *ptr = nullptr;
    PetscCall(chunk->release(stream));
    PetscCall(allocate(new_size, ptr, stream));
    PetscCall(allocator_.uninitialized_copy(*ptr, old_ptr, new_size, stream));
  } else {
    // chunk had enough room we can simply grow (or shrink) to fit the new size
    PetscCall(chunk->resize(new_size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace memory

} // namespace Petsc

#endif // PETSC_SEGMENTEDMEMPOOL_HPP
