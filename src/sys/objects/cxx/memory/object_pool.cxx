#include <petsc/private/cpp/object_pool.hpp>

#include <new>       // std::nothrow
#include <limits>    // std::numeric_limits
#include <algorithm> // std::lower_bound()
#include <cstdio>    // std::printf

namespace Petsc
{

namespace memory
{

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

struct PoolAllocator::AllocationHeader {
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
  PoolAllocator::AllocationHeader::AllocationHeader
*/
constexpr PoolAllocator::AllocationHeader::AllocationHeader(size_type size, align_type align) noexcept : size{size}, align{align} { }

/*
  PoolAllocator::AllocationHeader::max_alignment

  Returns the maximum supported alignment (in bytes) of the memory pool.
*/
constexpr PoolAllocator::align_type PoolAllocator::AllocationHeader::max_alignment() noexcept
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
constexpr PoolAllocator::size_type PoolAllocator::AllocationHeader::buffer_zone_size() noexcept
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
constexpr PoolAllocator::size_type PoolAllocator::AllocationHeader::header_size() noexcept
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
constexpr PoolAllocator::size_type PoolAllocator::total_size_(size_type size, align_type align) noexcept
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
PetscErrorCode PoolAllocator::delete_ptr_(void **in_ptr) noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(in_ptr, 1);
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
  PoolAllocator::find_align_ - Return an iterator to the memory pool for a particular alignment

  Input Parameter:
. align - The alignment (in bytes) to search for

  Notes:
  returns pool().end() if alignment not found.
*/
PoolAllocator::pool_type::iterator PoolAllocator::find_align_(align_type align) noexcept
{
  return std::lower_bound(this->pool().begin(), this->pool().end(), align, [](const pool_type::value_type &pair, const align_type &align) { return pair.first < align; });
}

PoolAllocator::pool_type::const_iterator PoolAllocator::find_align_(align_type align) const noexcept
{
  return std::lower_bound(this->pool().begin(), this->pool().end(), align, [](const pool_type::value_type &pair, const align_type &align) { return pair.first < align; });
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
PetscErrorCode PoolAllocator::clear_(size_type *remaining) noexcept
{
  size_type remain = 0;

  PetscFunctionBegin;
  if (remaining) PetscAssertPointer(remaining, 1);
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
  PetscCallCXX(this->pool().clear());
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
PetscErrorCode PoolAllocator::finalize_() noexcept
{
  PetscFunctionBegin;
  PetscCall(clear_());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// a quick sanity check that the alignment is valid, does nothing in optimized builds
PetscErrorCode PoolAllocator::valid_alignment_(align_type in_align) noexcept
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
  Setting check_in_ptr to false disabled the PetscAssertPointer() check in the function
  preamble. This allows the method to be used even if the aligned_ptr is poisoned (for example
  when extracting the header from a pointer that is checked into the pool).

  aligned_ptr must have been allocated by the pool.

  The returned header is still poisoned, the user is responsible for unpoisoning it.
*/
PetscErrorCode PoolAllocator::extract_header_(void *user_ptr, AllocationHeader **header, bool check_in_ptr) noexcept
{
  PetscFunctionBegin;
  if (check_in_ptr) PetscAssertPointer(user_ptr, 1);
  PetscAssertPointer(header, 2);
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
PetscErrorCode PoolAllocator::allocate_ptr_(size_type size, align_type align, void **ret_ptr) noexcept
{
  constexpr auto header_size = AllocationHeader::header_size();
  const auto     total_size  = total_size_(size, align);
  const auto     size_before = total_size - header_size;
  auto           usable_size = size_before;
  void          *aligned_ptr = nullptr;
  unsigned char *base_ptr    = nullptr;

  PetscFunctionBegin;
  PetscAssertPointer(ret_ptr, 1);
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
  base_ptr = ::new (std::nothrow) unsigned char[total_size];
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

PoolAllocator::~PoolAllocator() noexcept
{
  size_type leaked{};
  PetscBool init;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, clear_(&leaked));
  PetscCallAbort(PETSC_COMM_SELF, PetscInitialized(&init));
  if (init) PetscCheckAbort(leaked == 0, PETSC_COMM_SELF, PETSC_ERR_MEM_LEAK, "%zu objects remaining in the pool are leaked", leaked);
  PetscFunctionReturnVoid();
}

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
PetscErrorCode PoolAllocator::get_attributes(const void *ptr, size_type *size, align_type *align) noexcept
{
  PetscFunctionBegin;
  // ptr may be poisoned, so cannot check it here
  // PetscAssertPointer(out_ptr, 1);
  if (size) PetscAssertPointer(size, 2);
  if (align) PetscAssertPointer(align, 3);
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
PetscErrorCode PoolAllocator::try_allocate(void **out_ptr, size_type size, align_type align, bool *success) noexcept
{
  void *ptr{};
  bool  found{};

  PetscFunctionBegin;
  PetscAssertPointer(out_ptr, 1);
  PetscAssertPointer(success, 3);
  PetscCall(valid_alignment_(align));
  PetscCall(this->register_finalize());
  if (PetscLikely(size)) {
    const auto align_it = find_align_(align);

    if (align_it != this->pool().end() && align_it->first == align) {
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
PetscErrorCode PoolAllocator::allocate(void **out_ptr, size_type size, align_type align, bool *allocated_from_pool) noexcept
{
  bool success{};

  PetscFunctionBegin;
  PetscAssertPointer(out_ptr, 1);
  if (allocated_from_pool) PetscAssertPointer(allocated_from_pool, 3);
  PetscCall(try_allocate(out_ptr, size, align, &success));
  if (!success) PetscCall(allocate_ptr_(size, align, out_ptr));
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
PetscErrorCode PoolAllocator::deallocate(void **in_ptr, size_type size, align_type align) noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(in_ptr, 1);
  if (auto ptr = util::exchange(*in_ptr, nullptr)) {
    if (locked_) {
      // This is necessary if an object is "reclaimed" within another PetscFinalize()
      // registered cleanup after this pool has returned from its finalizer. In this case,
      // instead of pushing onto the stack we just delete the pointer directly.
      //
      // However this path is *only* valid if we have already finalized!
      PetscCall(delete_ptr_(&ptr));
    } else {
      auto it = find_align_(align);

      if (it == this->pool().end() || it->first != align) PetscCallCXX(it = this->pool().insert(it, {align, {}}));
      PetscCallCXX(it->second[size].emplace_back(ptr));
      PetscCall(PetscPoisonMemoryRegion(ptr, size));
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
PetscErrorCode PoolAllocator::unpoison(const void *ptr, size_type *size) noexcept
{
  PetscFunctionBegin;
  // ptr may be poisoned, so cannot check it here
  // PetscAssertPointer(ptr, 1);
  PetscAssertPointer(size, 2);
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
PetscErrorCode PoolAllocator::repoison(const void *ptr, size_type size) noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(ptr, 1);
  PetscCall(PetscPoisonMemoryRegion(ptr, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PoolAllocator::LockGuard PoolAllocator::lock_guard() noexcept
{
  return LockGuard{this};
}

// ==========================================================================================
// PoolAllocated -- Public API
// ==========================================================================================

void *PoolAllocated::operator new(size_type size) noexcept
{
  void *ptr{};

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, pool().allocate(&ptr, size, static_cast<align_type>(alignof(std::max_align_t))));
  PetscFunctionReturn(ptr);
}

void PoolAllocated::operator delete(void *ptr) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(ptr)) {
    size_type       size{};
    align_type      align{};
    allocator_type &allocated = pool();

    PetscCallAbort(PETSC_COMM_SELF, allocated.get_attributes(ptr, &size, &align));
    PetscCallAbort(PETSC_COMM_SELF, allocated.deallocate(&ptr, size, align));
  }
  PetscFunctionReturnVoid();
}

#if PETSC_CPP_VERSION >= 17
void *PoolAllocated::operator new(size_type size, std::align_val_t align) noexcept
{
  void *ptr{};

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, pool().allocate(&ptr, size, static_cast<align_type>(align)));
  PetscFunctionReturn(ptr);
}

void PoolAllocated::operator delete(void *ptr, std::align_val_t align) noexcept
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

// ==========================================================================================
// PoolAllocated -- Protected API
// ==========================================================================================

PoolAllocated::allocator_type &PoolAllocated::pool() noexcept
{
  return pool_;
}

} // namespace memory

} // namespace Petsc
