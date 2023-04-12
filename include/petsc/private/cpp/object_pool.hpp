#ifndef PETSC_CPP_OBJECT_POOL_HPP
#define PETSC_CPP_OBJECT_POOL_HPP

#include <petsc/private/petscimpl.h> // PetscValidPointer()
#include <petsc/private/mempoison.h> // PetscPoison/UnpoisonMemoryRegion()

#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/utility.hpp> // util::exchange()
#include <petsc/private/cpp/unordered_map.hpp>
#include <petsc/private/cpp/memory.hpp> // std::align()

#if defined(__cplusplus)
  #include <cstddef> // std::max_align_t
  #include <limits>  // std::numeric_limits
  #include <deque>   // std::take_a_wild_guess
  #include <new>     // std::nothrow

namespace Petsc
{

namespace memory
{

enum class align_val_t : std::size_t {
};

} // namespace memory

} // namespace Petsc

namespace std
{

template <>
struct hash<::Petsc::memory::align_val_t> {
  #if PETSC_CPP_VERSION < 17
  using argument_type = ::Petsc::memory::align_val_t;
  using result_type   = size_t;
  #endif

  constexpr size_t operator()(const ::Petsc::memory::align_val_t &x) const noexcept { return static_cast<size_t>(x); }
};

} // namespace std

namespace Petsc
{

namespace memory
{

// ==========================================================================================
// PoolAllocator
//
// A general purpose memory pool. It internally maintains an array of allocated memory regions
// and their sizes. Currently does not prune the allocated memory in any way.
// ==========================================================================================

class PoolAllocator : public RegisterFinalizeable<PoolAllocator> {
  using base_type = RegisterFinalizeable<PoolAllocator>;
  friend base_type;

public:
  // define the size and alignment as separate types, this helps to disambiguate them at the
  // callsite!
  using size_type  = std::size_t;
  using align_type = align_val_t;
  using pool_type  = UnorderedMap<align_type, UnorderedMap<size_type, std::deque<void *>>>;

  PoolAllocator() noexcept                            = default;
  PoolAllocator(PoolAllocator &&) noexcept            = default;
  PoolAllocator &operator=(PoolAllocator &&) noexcept = default;

  // the pool carries raw memory and is not copyable
  PoolAllocator(const PoolAllocator &)            = delete;
  PoolAllocator &operator=(const PoolAllocator &) = delete;

  #if PetscDefined(USE_DEBUG)
  // in debug mode we check that we haven't leaked anything
  ~PoolAllocator() noexcept
  {
    size_type leaked = 0;

    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, clear_(&leaked));
    PetscCheckAbort(leaked == 0, PETSC_COMM_SELF, PETSC_ERR_MEM, "%zu objects remaining in the pool are leaked", leaked);
    PetscFunctionReturnVoid();
  }
  #endif

  PetscErrorCode        try_allocate(void **, size_type, align_type, bool *) noexcept;
  PetscErrorCode        allocate(void **, size_type, align_type, bool        * = nullptr) noexcept;
  PetscErrorCode        deallocate(void **, size_type, align_type) noexcept;
  static PetscErrorCode get_attributes(const void *, size_type *, align_type *) noexcept;

  static PetscErrorCode unpoison(const void *, size_type *) noexcept;
  static PetscErrorCode repoison(const void *, size_type) noexcept;

  template <typename T>
  PetscErrorCode for_each(T &&) noexcept;

  PETSC_NODISCARD pool_type       &pool() noexcept { return pool_; }
  PETSC_NODISCARD const pool_type &pool() const noexcept { return pool_; }

private:
  pool_type pool_{};

  class AllocationHeader;

  PETSC_NODISCARD static constexpr size_type total_size_(size_type, align_type) noexcept;

  static PetscErrorCode valid_alignment_(align_type) noexcept;
  static PetscErrorCode extract_header_(void *, AllocationHeader **, bool = true) noexcept;
  static PetscErrorCode allocate_ptr_(size_type, align_type, void **) noexcept;

  static PetscErrorCode delete_ptr_(void **) noexcept;

  PetscErrorCode clear_(size_type * = nullptr) noexcept;
  PetscErrorCode finalize_() noexcept;
};

// ==========================================================================================
// PoolAllocator -- Private API -- AllocationHeader
//
// The header inserted for each allocated pointer. It stores:
//
// - size - the size (in bytes) of the allocation. This includes ONLY the size as requested by
//          the user. i.e. if a user requested 10 bytes but alignment, padding and header
//          overhead results in the actual allocation being 30 bytes, then size = 10.
// - align - the alignment (in bytes) of the allocated pointer.
// ==========================================================================================

class PoolAllocator::AllocationHeader {
public:
  constexpr AllocationHeader(size_type, align_type) noexcept;

  PETSC_NODISCARD static constexpr align_type max_alignment() noexcept;
  PETSC_NODISCARD static constexpr size_type  header_size() noexcept;
  PETSC_NODISCARD static constexpr size_type  buffer_zone_size() noexcept;

  size_type  size;
  align_type align;
};

// ==========================================================================================
// PoolAllocator -- Private API -- AllocationHeader -- Public API
// ==========================================================================================

/*
  PoolAlocator::AllocationHeader::AllocationHeader
*/
inline constexpr PoolAllocator::AllocationHeader::AllocationHeader(size_type size, align_type align) noexcept : size(size), align(align) { }

/*
  PoolAllocator::AllocationHeader::max_alignment

  Returns the maximum supported alignment (in bytes) of the memory pool.
*/
inline constexpr PoolAllocator::align_type PoolAllocator::AllocationHeader::max_alignment() noexcept
{
  #if PETSC_CPP_VERSION >= 14
  constexpr auto max_align = std::numeric_limits<unsigned char>::max() + 1;
  static_assert(!(max_align & (max_align - 1)), "Maximum alignment must be a power of 2");
  return static_cast<align_type>(max_align);
  #else
  return static_cast<align_type>(std::numeric_limits<unsigned char>::max() + 1);
  #endif
}

/*
  PoolAllocator::AllocationHeader::buffer_zone_size

  Notes:
  Returns the number of bytes between the allocated pointer and the location where the
  alignment diff is stored. i.e. size of the buffer zone + 1.

  If ASAN is enabled then this buffer zone is poisoned, so any overrun on part of the user is
  potentially caught. The larger the buffer zone, the more likely that the user lands in
  poisoned memory. Turned off in optimized builds.
*/
inline constexpr PoolAllocator::size_type PoolAllocator::AllocationHeader::buffer_zone_size() noexcept
{
  return (PetscDefined(USE_DEBUG) ? 32 : 0) + 1;
}

/*
  PoolAllocator::AllocationHeader::header_size

  Notes:
  Returns the minimum size of the allocation header in bytes. Essentially (literally) the size
  of the header object + size of the buffer zone. Does not include padding due to alignment
  offset itself though.
*/
inline constexpr PoolAllocator::size_type PoolAllocator::AllocationHeader::header_size() noexcept
{
  return sizeof(AllocationHeader) + buffer_zone_size();
}

/*
  PoolAllocator::total_size_ - Compute the maximum total size for an allocation

  Input Parameters:
+ size  - the size (in bytes) requested by the user
- align - the alignment (in bytes) requested by the user

  Notes:
  Returns a size so that std::malloc(total_size_(size, align)) allocates enough memory to store
  the allocation header, buffer zone, alignment offset and requested size including any
  potential changes due to alignment.
*/
inline constexpr PoolAllocator::size_type PoolAllocator::total_size_(size_type size, align_type align) noexcept
{
  // align - 1 because aligning the pointer up by align bytes just returns it to its original
  // alignment, so we can save the byte
  return AllocationHeader::header_size() + size + util::to_underlying(align) - 1;
}

/*
  PoolAllocator::delete_ptr_ - deletes a pointer and the corresponding header

  Input Parameter:
+ in_ptr - the pointer to user-memory which to delete

  Notes:
  in_ptr may point to nullptr (in which case this does nothing), otherwise it must have been
  allocated by the pool.

  in_ptr may point to poisoned memory, this routine will remove any poisoning before
  deallocating.

  in_ptr is set to nullptr on return.
*/
inline PetscErrorCode PoolAllocator::delete_ptr_(void **in_ptr) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(in_ptr, 1);
  if (const auto ptr = util::exchange(*in_ptr, nullptr)) {
    AllocationHeader *header = nullptr;

    PetscCall(extract_header_(ptr, &header, false));
    // must unpoison the header itself before we can access the members
    PetscCall(PetscUnpoisonMemoryRegion(header, sizeof(*header)));
    PetscCall(PetscUnpoisonMemoryRegion(header, total_size_(header->size, header->align)));
    PetscCallCXX(::delete[] reinterpret_cast<unsigned char *>(header));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::clear_ - Clear the memory pool

  Output Parameter:
. remaining - The number of remaining allocations in the pool, nullptr if not needed

  Notes:
  This will clean up the pool, deallocating any memory checked back into the pool. This does
  not delete allocations that were allocated by the pool but not yet returned to it.

  remaining is useful in determining if any allocations were "leaked". Suppose an object
  internally manages a resource and uses the pool to allocate said resource. On destruction the
  object expects to have the pool be empty, i.e. have remaining = 0. This implies all resources
  were returned to the pool.
*/
inline PetscErrorCode PoolAllocator::clear_(size_type *remaining) noexcept
{
  size_type remain = 0;

  PetscFunctionBegin;
  if (remaining) PetscValidPointer(remaining, 1);
  // clang-format off
  PetscCall(
    this->for_each([&](void *&ptr)
    {
      PetscFunctionBegin;
      ++remain;
      PetscCall(delete_ptr_(&ptr));
      PetscFunctionReturn(PETSC_SUCCESS);
    })
  );
  // clang-format on
  PetscCall(pool().clear());
  if (remaining) *remaining = remain;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::finalize_ - Routine automatically called during PetscFinalize()

  Notes:
  This will go through and reap any regions that it owns in the underlying container. If it
  owns all regions, it resets the container.

  There currently is no way to ensure that objects remaining in the pool aren't leaked, since
  this routine cannot actually re-register the pool for finalizations without causing an
  infinite loop...

  Thus it is up to the owned object to ensure that the pool is properly finalized.
*/
inline PetscErrorCode PoolAllocator::finalize_() noexcept
{
  PetscFunctionBegin;
  PetscCall(clear_());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// a quick sanity check that the alignment is valid, does nothing in optimized builds
inline PetscErrorCode PoolAllocator::valid_alignment_(align_type in_align) noexcept
{
  constexpr auto max_align = util::to_underlying(AllocationHeader::max_alignment());
  const auto     align     = util::to_underlying(in_align);

  PetscFunctionBegin;
  PetscAssert((align > 0) && (align <= max_align), PETSC_COMM_SELF, PETSC_ERR_MEMC, "Alignment %zu must be (0, %zu]", align, max_align);
  PetscAssert(!(align & (align - 1)), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Alignment %zu must be a power of 2", align);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::extract_header_ - Extract the header pointer from the aligned pointer

  Input Parameters:
+ user_ptr     - the pointer to the aligned user memory
- check_in_ptr - whether to test the validity of aligned_ptr

  Output Parameter:
. header - the pointer to the header

  Notes:
  Setting check_in_ptr to false disabled the PetscValidPointer() check in the function
  preamble. This allows the method to be used even if the aligned_ptr is poisoned (for example
  when extracting the header from a pointer that is checked into the pool).

  aligned_ptr must have been allocated by the pool.

  The returned header is still poisoned, the user is responsible for unpoisoning it.
*/
inline PetscErrorCode PoolAllocator::extract_header_(void *user_ptr, AllocationHeader **header, bool check_in_ptr) noexcept
{
  PetscFunctionBegin;
  if (check_in_ptr) PetscValidPointer(user_ptr, 1);
  PetscValidPointer(header, 2);
  {
    //       AllocationHeader::alignment_offset() (at least 1)
    //                        |
    // header                 |  user_ptr/aligned_ptr
    // |                      |          |
    // v                  v~~~~~~~~~~~~~~v
    // A==============B===C==============D--------------------- ...
    // ^~~~~~~~~~~~~~~^^~~^              ^~~~~~~~~~~~~~~~~~~~~~ ...
    //       |             \______             user memory
    //       |                    buffer_zone_end
    // sizeof(AllocationHeader)
    //
    // ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
    //                poisoned
    //
    const auto aligned_ptr     = reinterpret_cast<unsigned char *>(user_ptr);
    const auto buffer_zone_end = aligned_ptr - AllocationHeader::buffer_zone_size();

    PetscCall(PetscUnpoisonMemoryRegion(buffer_zone_end, sizeof(*buffer_zone_end)));
    {
      // offset added to original pointer due to alignment, B -> C above (may be zero)
      const auto alignment_offset = *buffer_zone_end;

      *header = reinterpret_cast<AllocationHeader *>(buffer_zone_end - alignment_offset - sizeof(AllocationHeader));
    }
    PetscCall(PetscPoisonMemoryRegion(buffer_zone_end, sizeof(*buffer_zone_end)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::allocate_ptr_ - Allocate a pointer and header given requested size and
  alignment

  Input Parameters:
+ size  - the size (in bytes) to allocate
- align - the size (in bytes) to align the pointer to

  Output Parameter:
. ret_ptr - the resulting pointer to user-memory

  Notes:
  Both size and align must be > 0. align must be a power of 2.
  This both allocates the user memory and the corresponding metadata region.
*/
inline PetscErrorCode PoolAllocator::allocate_ptr_(size_type size, align_type align, void **ret_ptr) noexcept
{
  constexpr auto header_size = AllocationHeader::header_size();
  const auto     total_size  = total_size_(size, align);
  const auto     size_before = total_size - header_size;
  auto           usable_size = size_before;
  void          *aligned_ptr = nullptr;
  unsigned char *base_ptr    = nullptr;

  PetscFunctionBegin;
  PetscValidPointer(ret_ptr, 1);
  PetscCall(valid_alignment_(align));
  // memory is laid out as follows:
  //
  //                            aligned_ptr     ret_ptr (and aligned_ptr after std::align())
  // base_ptr     buffer_zone       |     _____/
  // |                      |       |    /     user memory
  // v                  v~~~~~~~~~~~x~~~v~~~~~~~~~~~~~~~~~~~~~ ...
  // A==============B===C===========D===E--------------------- ...
  // ^~~~~~~~~~~~~~~^^~~^
  //        |            \_________
  // sizeof(AllocationHeader)      |
  //                               alignment_offset (may be 0)
  //
  // ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
  //                poisoned
  //                                ^~~~~~~~~~~~~~~~~~~~~~~~~~ ...
  //                                          usable_size
  // ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ...
  //                       total_size_()
  //
  base_ptr = ::new (std::nothrow_t{}) unsigned char[total_size];
  PetscAssert(base_ptr, PETSC_COMM_SELF, PETSC_ERR_MEM, "operator new() failed to allocate %zu bytes", total_size);
  PetscCallCXX(base_ptr = reinterpret_cast<unsigned char *>(util::construct_at(reinterpret_cast<AllocationHeader *>(base_ptr), size, align)));
  aligned_ptr = base_ptr + header_size;
  // storing to ret_ptr and not aligned_ptr is deliberate! std::align() returns nullptr if it
  // fails, so we do not want to clobber aligned_ptr
  *ret_ptr = std::align(util::to_underlying(align), size, aligned_ptr, usable_size);
  // note usable_size is has now shrunk by alignment_offset
  PetscAssert(*ret_ptr, PETSC_COMM_SELF, PETSC_ERR_LIB, "std::align() failed to align pointer %p (size %zu, alignment %zu)", aligned_ptr, size, util::to_underlying(align));
  {
    constexpr auto max_align        = util::to_underlying(AllocationHeader::max_alignment());
    const auto     alignment_offset = size_before - usable_size;

    PetscAssert(alignment_offset <= max_align, PETSC_COMM_SELF, PETSC_ERR_MEMC, "Computed alignment offset %zu > maximum allowed alignment %zu", alignment_offset, max_align);
    *(reinterpret_cast<unsigned char *>(aligned_ptr) - AllocationHeader::buffer_zone_size()) = static_cast<unsigned char>(alignment_offset);
    if (PetscDefined(USE_DEBUG)) {
      const auto computed_aligned_ptr = base_ptr + header_size + alignment_offset;

      PetscCheck(computed_aligned_ptr == aligned_ptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Base pointer %p + header size %zu + alignment offset %zu = %p != aligned pointer %p", static_cast<void *>(base_ptr), header_size, alignment_offset, static_cast<void *>(computed_aligned_ptr), aligned_ptr);
    }
  }
  // Poison the entire region first, then unpoison only the user region. This ensures that
  // any extra space on *either* ends of the array are poisoned
  PetscCall(PetscPoisonMemoryRegion(base_ptr, total_size));
  PetscCall(PetscUnpoisonMemoryRegion(aligned_ptr, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// PoolAllocator -- Public API
// ==========================================================================================

/*
  PoolAllocator::get_attributes - Get the size and alignment of an allocated pointer

  Input Parameter:
. ptr - the pointer to query

  Output Parameters:
+ size - the size (in bytes) of the allocated area, nullptr if not needed
- align - the alignment (in bytes) of the allocated, nullptr if not needed

  Note:
  ptr must have been allocated by the pool, and is exactly the pointer returned by either
  allocate() or try_allocate() (if successful).
*/
inline PetscErrorCode PoolAllocator::get_attributes(const void *ptr, size_type *size, align_type *align) noexcept
{
  PetscFunctionBegin;
  // ptr may be poisoned, so cannot check it here
  // PetscValidPointer(out_ptr, 1);
  if (size) PetscValidPointer(size, 2);
  if (align) PetscValidPointer(align, 3);
  if (PetscLikely(size || align)) {
    AllocationHeader *header = nullptr;

    PetscCall(extract_header_(const_cast<void *>(ptr), &header, /* check ptr = */ false));
    PetscCall(PetscUnpoisonMemoryRegion(header, sizeof(*header)));
    if (size) *size = header->size;
    if (align) *align = header->align;
    PetscCall(PetscPoisonMemoryRegion(header, sizeof(*header)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::try_allocate - Attempt to allocate memory from the pool

  Input Parameter:
+ size  - the size (in bytes) to attempt to allocate
- align - the alignment (in bytes) of the requested allocation

  Output Parameters:
+ out_ptr - the pointer to return the allocated memory in
- success - set to true if out_ptr was successfully is allocated

  Notes:
  Differs from allocate() insofar that this routine does not allocate new memory if it does not
  find a suitable memory chunk in the pool.

  align must be a power of 2, and > 0.

  If size is 0, out_ptr is set to nullptr and success set to false
*/
inline PetscErrorCode PoolAllocator::try_allocate(void **out_ptr, size_type size, align_type align, bool *success) noexcept
{
  void *ptr   = nullptr;
  auto  found = false;

  PetscFunctionBegin;
  PetscValidPointer(out_ptr, 1);
  PetscValidPointer(success, 3);
  PetscCall(valid_alignment_(align));
  if (PetscLikely(size)) {
    const auto align_it = pool().find(align);

    if (align_it != pool().end()) {
      auto     &&size_map = align_it->second;
      const auto size_it  = size_map.find(size);

      if (size_it != size_map.end()) {
        auto &&ptr_list = size_it->second;

        if (!ptr_list.empty()) {
          found = true;
          ptr   = ptr_list.back();
          PetscCallCXX(ptr_list.pop_back());
          PetscCall(PetscUnpoisonMemoryRegion(ptr, size));
        }
      }
    }
  }
  *out_ptr = ptr;
  *success = found;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::allocate - Allocate a chunk of memory from the pool

  Input Parameters:
+ size  - The size (in bytes) to allocate
- align - The alignment (in bytes) to align the allocation to

  Output Parameters:
+ out_ptr             - A pointer containing the beginning of the allocated region
- allocated_from_pool - True if the region was allocated from the pool, false otherwise

  Notes:
  If size is 0, out_ptr is set to nullptr and was_allocated is set to false.
*/
inline PetscErrorCode PoolAllocator::allocate(void **out_ptr, size_type size, align_type align, bool *allocated_from_pool) noexcept
{
  bool success;

  PetscFunctionBegin;
  PetscValidPointer(out_ptr, 1);
  if (allocated_from_pool) PetscValidPointer(allocated_from_pool, 3);
  PetscCall(try_allocate(out_ptr, size, align, &success));
  if (!success) {
    PetscCall(this->register_finalize());
    PetscCall(allocate_ptr_(size, align, out_ptr));
  }
  if (allocated_from_pool) *allocated_from_pool = success;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocate::deallocate - Return a pointer to the pool

  Input Parameter:
. in_ptr - A pointer to the beginning of the allocated region

  Notes:
  On success the region pointed to by in_ptr is poisoned. Any further attempts to access
  the memory pointed to by in_ptr will result in an error.

  in_ptr must have been allocated by the pool, and must point to the beginning of the allocated
  region.

  The value in_ptr points to may be nullptr, in which case this routine does nothing.
*/
inline PetscErrorCode PoolAllocator::deallocate(void **in_ptr, size_type size, align_type align) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(in_ptr, 1);
  if (auto ptr = util::exchange(*in_ptr, nullptr)) {
    if (this->registered()) {
      PetscCallCXX(pool()[align][size].emplace_back(ptr));
      PetscCall(PetscPoisonMemoryRegion(ptr, size));
    } else {
      // This is necessary if an object is "reclaimed" within another PetscFinalize()
      // registered cleanup after this pool has returned from its finalizer. In this case,
      // instead of pushing onto the stack we just delete the pointer directly.
      //
      // However this path is *only* valid if we have already finalized!
      PetscCall(delete_ptr_(&ptr));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::unpoison - Unpoison a pool-allocated pointer

  Input Parameter:
. ptr - the pointer to poison

  Output Parameter:
. size - the size (in bytes) of the region pointed to by ptr

  Notes:
  ptr must not be nullptr.

  size should be passed to the corresponding repoison() to undo the effects of this
  routine.

  Using this routine in conjunction with unpoison() allows a user to temporarily push and pop
  the poisoning state of a given pointer. The pool does not repoison the pointer for you, so
  use at your own risk!
*/
inline PetscErrorCode PoolAllocator::unpoison(const void *ptr, size_type *size) noexcept
{
  PetscFunctionBegin;
  // ptr may be poisoned, so cannot check it here
  // PetscValidPointer(ptr, 1);
  PetscValidPointer(size, 2);
  PetscCall(get_attributes(ptr, size, nullptr));
  PetscCall(PetscUnpoisonMemoryRegion(ptr, *size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::repoison - Poison a pointer previously unpoisoned via unpoison()

  Input Parameters:
+ ptr  - the pointer to the unpoisoned region
- size - the size of the region

  Notes:
  size must be exactly the value returned by unpoison().

  ptr cannot be nullptr
*/
inline PetscErrorCode PoolAllocator::repoison(const void *ptr, size_type size) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(ptr, 1);
  PetscCall(PetscPoisonMemoryRegion(ptr, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PoolAllocator::for_each - Perform an action on every allocation in the currently in the pool

  Input Parameter:
. callable - The callable used to perform the action, should accept a void *&.

  Notes:
  The callable may delete the pointer, but MUST set the pointer to nullptr in this case. If the
  pointer is deleted, it is removed from the pool.

  The pointers are walked in LIFO order from most recent first to least recent deallocation
  last.
*/
template <typename T>
inline PetscErrorCode PoolAllocator::for_each(T &&callable) noexcept
{
  PetscFunctionBegin;
  for (auto &&align_it : pool()) {
    for (auto &&size_it : align_it.second) {
      auto &&ptr_stack = size_it.second;

      for (auto it = ptr_stack.rbegin(); it != ptr_stack.rend();) {
        size_type size;
        auto    &&ptr = *it;

        PetscCall(unpoison(ptr, &size));
        PetscCall(callable(ptr));
        if (ptr) {
          PetscCall(repoison(ptr, size));
          ++it;
        } else {
          // the callable has deleted the pointer, so we should remove it. it is a reverse
          // iterator though so we:
          //
          // 1. std::next(it).base() -> convert to forward iterator
          // 2. it = decltype(it){...} -> convert returned iterator back to reverse iterator
          PetscCallCXX(it = decltype(it){ptr_stack.erase(std::next(it).base())});
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// PoolAllocated
//
// A simple mixin to enable allocating a class from a Pool. That is, it provides a static
// operator new() and operator delete() member functions such that
//
// auto foo = new ClassDerivedFromPoolAllocated(args...);
//
// Allocates the new object from a pool and
//
// delete foo;
//
// Returns the memory to the pool.
// ==========================================================================================

template <typename Derived>
class PoolAllocated {
public:
  using allocator_type = PoolAllocator;
  using size_type      = typename allocator_type::size_type;
  using align_type     = typename allocator_type::align_type;

  PETSC_NODISCARD static void *operator new(size_type) noexcept;
  static void operator delete(void *) noexcept;

  #if PETSC_CPP_VERSION >= 17
  PETSC_NODISCARD static void *operator new(size_type, std::align_val_t) noexcept;
  static void operator delete(void *, std::align_val_t) noexcept;
  #endif

protected:
  PETSC_NODISCARD static allocator_type &pool() noexcept;

private:
  static allocator_type pool_;
};

template <typename D>
typename PoolAllocated<D>::allocator_type PoolAllocated<D>::pool_{};

template <typename D>
inline void *PoolAllocated<D>::operator new(size_type size) noexcept
{
  void *ptr = nullptr;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, pool().allocate(&ptr, size, static_cast<align_type>(alignof(std::max_align_t))));
  PetscFunctionReturn(ptr);
}

template <typename D>
inline void PoolAllocated<D>::operator delete(void *ptr) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(ptr)) {
    size_type  size{};
    align_type align{};

    PetscCallAbort(PETSC_COMM_SELF, pool().get_attributes(ptr, &size, &align));
    PetscCallAbort(PETSC_COMM_SELF, pool().deallocate(&ptr, size, align));
  }
  PetscFunctionReturnVoid();
}

  #if PETSC_CPP_VERSION >= 17
template <typename D>
inline void *PoolAllocated<D>::operator new(size_type size, std::align_val_t align) noexcept
{
  void *ptr = nullptr;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, pool().allocate(&ptr, size, static_cast<align_type>(align)));
  PetscFunctionReturn(ptr);
}

template <typename D>
inline void PoolAllocated<D>::operator delete(void *ptr, std::align_val_t align) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(ptr)) {
    size_type size{};

    PetscCallAbort(PETSC_COMM_SELF, pool().get_attributes(ptr, &size, nullptr));
    PetscCallAbort(PETSC_COMM_SELF, pool().deallocate(&ptr, size, static_cast<align_type>(align)));
  }
  PetscFunctionReturnVoid();
}
  #endif

template <typename D>
inline typename PoolAllocated<D>::allocator_type &PoolAllocated<D>::pool() noexcept
{
  return pool_;
}

} // namespace memory

// ==========================================================================================
// ConstructorInterface
//
// Provides a common interface for constructors and destructors for use with the object
// pools. Specifically, each interface may provide the following functions:
//
// construct_(T *):
// Given a pointer to an allocated object, construct the object in that memory. This may
// allocate memory *within* the object, but must not reallocate the pointer itself. Defaults to
// placement new.
//
// destroy_(T *):
// Given a pointer to an object, destroy the object completely. This should clean up the
// pointed-to object but not deallocate the pointer itself. Any resources not cleaned up by
// this function will be leaked. Defaults to calling the objects destructor.
//
// invalidate_(T *):
// Similar to destroy_(), but you are allowed to leave resources behind in the
// object. Essentially puts the object into a "zombie" state to be reused later. Defaults to
// calling destroy_().
//
// reset_():
// Revives a previously invalidated object. Should restore the object to a "factory new" state
// as-if it had just been newly allocated, but may take advantage of the fact that some
// resources need not be re-aquired from scratch. Defaults to calling construct_().
// ==========================================================================================

template <typename T, typename Derived>
class ConstructorInterface {
public:
  using value_type = T;

  template <typename... Args>
  PetscErrorCode construct(value_type *ptr, Args &&...args) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().construct_(ptr, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode destroy(value_type *ptr) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().destroy_(ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename... Args>
  PetscErrorCode reset(value_type *val, Args &&...args) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().reset_(val, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode invalidate(value_type *ptr) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().invalidate_(ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

protected:
  template <typename... Args>
  static PetscErrorCode construct_(value_type *ptr, Args &&...args) noexcept
  {
    PetscFunctionBegin;
    PetscValidPointer(ptr, 1);
    PetscCallCXX(util::construct_at(ptr, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode destroy_(value_type *ptr) noexcept
  {
    PetscFunctionBegin;
    if (ptr) {
      PetscValidPointer(ptr, 1);
      PetscCallCXX(util::destroy_at(ptr));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename... Args>
  PetscErrorCode reset_(value_type *val, Args &&...args) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().construct(val, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode invalidate_(value_type *ptr) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().destroy(ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PETSC_NODISCARD Derived       &underlying() noexcept { return static_cast<Derived &>(*this); }
  PETSC_NODISCARD const Derived &underlying() const noexcept { return static_cast<const Derived &>(*this); }
};

template <typename T>
struct DefaultConstructor : ConstructorInterface<T, DefaultConstructor<T>> { };

// ==========================================================================================
// ObjectPool
//
// multi-purpose basic object-pool, useful for recirculating old "destroyed" objects. Registers
// all objects to be cleaned up on PetscFinalize()
// ==========================================================================================

template <typename T, typename Constructor = DefaultConstructor<T>>
class ObjectPool;

template <typename T, typename Constructor>
class ObjectPool : public RegisterFinalizeable<ObjectPool<T, Constructor>> {
  using base_type = RegisterFinalizeable<ObjectPool<T, Constructor>>;

public:
  using value_type       = T;
  using constructor_type = Constructor;
  using allocator_type   = memory::PoolAllocator;

  ObjectPool()                                  = default;
  ObjectPool(ObjectPool &&) noexcept            = default;
  ObjectPool &operator=(ObjectPool &&) noexcept = default;

  ~ObjectPool() noexcept;

  template <typename... Args>
  PetscErrorCode allocate(value_type **, Args &&...) noexcept;
  PetscErrorCode deallocate(value_type **) noexcept;

  PETSC_NODISCARD constructor_type       &constructor() noexcept { return pair_.first(); }
  PETSC_NODISCARD const constructor_type &constructor() const noexcept { return pair_.first(); }
  PETSC_NODISCARD allocator_type         &allocator() noexcept { return pair_.second(); }
  PETSC_NODISCARD const allocator_type   &allocator() const noexcept { return pair_.second(); }

private:
  util::compressed_pair<constructor_type, allocator_type> pair_{};

  using align_type = typename allocator_type::align_type;
  using size_type  = typename allocator_type::size_type;

  friend base_type;
  PetscErrorCode finalize_() noexcept;
};

// ==========================================================================================
// ObjectPool -- Private API
// ==========================================================================================

template <typename T, typename Constructor>
inline PetscErrorCode ObjectPool<T, Constructor>::finalize_() noexcept
{
  PetscFunctionBegin;
  // clang-format off
  PetscCall(
    this->allocator().for_each([&, this](void *ptr)
    {
      PetscFunctionBegin;
      PetscCall(this->constructor().destroy(static_cast<value_type *>(ptr)));
      PetscFunctionReturn(PETSC_SUCCESS);
    })
  );
  // clang-format on
  PetscCall(this->allocator().register_finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// ObjectPool -- Public API
// ==========================================================================================

template <typename T, typename Constructor>
inline ObjectPool<T, Constructor>::~ObjectPool() noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, this->finalize());
  PetscFunctionReturnVoid();
}

/*
  ObjectPool::allocate - Allocate an object from the pool

  Input Parameters:
. args... - The arguments to be passed to the constructor for the object

  Output Parameter:
. obj - The pointer to the constructed object

  Notes:
  The user must deallocate the object using deallocate() and cannot free it themselves.
*/
template <typename T, typename Constructor>
template <typename... Args>
inline PetscErrorCode ObjectPool<T, Constructor>::allocate(value_type **obj, Args &&...args) noexcept
{
  auto  allocated_from_pool = true;
  void *mem                 = nullptr;

  PetscFunctionBegin;
  PetscValidPointer(obj, 1);
  // order is deliberate! We register our finalizer before the pool does so since we need to
  // destroy the objects within it before it deletes their memory.
  PetscCall(this->register_finalize());
  PetscCall(this->allocator().allocate(&mem, sizeof(value_type), static_cast<align_type>(alignof(value_type)), &allocated_from_pool));
  *obj = static_cast<value_type *>(mem);
  if (allocated_from_pool) {
    // if the allocation reused memory from the pool then this indicates the object is resettable.
    PetscCall(this->constructor().reset(*obj, std::forward<Args>(args)...));
  } else {
    PetscCall(this->constructor().construct(*obj, std::forward<Args>(args)...));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ObjectPool::deallocate - Return an object to the pool

  Input Parameter:
. obj - The pointer to the object to return

  Notes:
  Sets obj to nullptr on return. obj must have been allocated by the pool in order to be
  deallocated this way.
*/
template <typename T, typename Constructor>
inline PetscErrorCode ObjectPool<T, Constructor>::deallocate(value_type **obj) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(obj, 1);
  if (this->registered()) {
    PetscCall(this->constructor().invalidate(*obj));
  } else {
    PetscCall(this->constructor().destroy(*obj));
  }
  PetscCall(this->allocator().deallocate(reinterpret_cast<void **>(obj), sizeof(value_type), static_cast<align_type>(alignof(value_type))));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_OBJECT_POOL_HPP
