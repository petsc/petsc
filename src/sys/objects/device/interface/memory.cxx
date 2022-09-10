#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/

#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/array.hpp>

#include <unordered_map>
#include <algorithm> // std::find_if
#include <cstring>   // std::memset

const char *const PetscDeviceCopyModes[] = {"host_to_host", "device_to_host", "host_to_device", "device_to_device", "auto", "PetscDeviceCopyMode", "PETSC_DEVICE_COPY_", nullptr};
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOH) == 0, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOH) == 1, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_HTOD) == 2, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_DTOD) == 3, "");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_COPY_AUTO) == 4, "");

// ==========================================================================================
// MemoryMap
//
// Since the pointers allocated via PetscDeviceAllocate() may be device pointers we cannot just
// store meta-data within the pointer itself (as we can't dereference them). So instead we need
// to keep an extra map to keep track of them
//
// Each entry maps pointer -> [PetscMemType, PetscObjectId, size]
// ==========================================================================================

// GCC implementation for std::hash<T*>. LLVM's libc++ is almost 2x slower because they do all
// kinds of complicated murmur hashing, so we make sure to enforce GCC's version.
struct PointerHash {
  std::size_t operator()(const void *ptr) const noexcept { return reinterpret_cast<std::size_t>(ptr); }
};

class MemoryMap : public Petsc::RegisterFinalizeable<MemoryMap> {
public:
  struct PointerAttributes {
    PetscMemType  mtype{}; // memtype of allocation
    PetscObjectId id{};    // id of allocation
    std::size_t   size{};  // size of allocation (bytes)

    // even though this is a POD and can be aggregate initialized, the STL uses () constructors
    // in unordered_map and so we need to provide a trivial contructor...
    constexpr PointerAttributes(PetscMemType, PetscObjectId, std::size_t) noexcept;
    constexpr PointerAttributes() noexcept                                              = default;
    constexpr PointerAttributes(const PointerAttributes &) noexcept                     = default;
    PETSC_CONSTEXPR_14 PointerAttributes &operator=(const PointerAttributes &) noexcept = default;
    constexpr PointerAttributes(PointerAttributes &&) noexcept                          = default;
    PETSC_CONSTEXPR_14 PointerAttributes &operator=(PointerAttributes &&) noexcept      = default;

    bool operator==(const PointerAttributes &) const noexcept;

    PETSC_NODISCARD bool contains(const void *, const void *) const noexcept;
  };

  using map_type = std::unordered_map<void *, PointerAttributes, PointerHash>;

  map_type map;

  // return the iterator of the allocation containing ptr, or map.cend() if not found
  PETSC_NODISCARD map_type::const_iterator search_for(const void *) const noexcept;

private:
  friend class Petsc::RegisterFinalizeable<MemoryMap>;
  PETSC_NODISCARD PetscErrorCode register_finalize_() noexcept;
  PETSC_NODISCARD PetscErrorCode finalize_() noexcept;
};

// ==========================================================================================
// PointerAttributes
// ==========================================================================================

constexpr MemoryMap::PointerAttributes::PointerAttributes(PetscMemType mtype_, PetscObjectId id_, std::size_t size_) noexcept : mtype(mtype_), id(id_), size(size_) { }

bool MemoryMap::PointerAttributes::operator==(const PointerAttributes &other) const noexcept {
  return mtype == other.mtype && id == other.id && size == other.size;
}

bool MemoryMap::PointerAttributes::contains(const void *ptr_begin, const void *ptr) const noexcept {
  return (ptr >= ptr_begin) && (ptr < (static_cast<const char *>(ptr_begin) + size));
}

// ==========================================================================================
// Memory map functions
// ==========================================================================================

PetscErrorCode MemoryMap::register_finalize_() noexcept {
  PetscFunctionBegin;
  // Preallocate, this does give a modest performance bump since unordered_map is so __dog__
  // slow if it needs to rehash. Experiments show that users tend not to have more than 5 or
  // so concurrently live pointers lying around. 10 at most.
  PetscCallCXX(map.reserve(16));
  PetscFunctionReturn(0);
}

PetscErrorCode MemoryMap::finalize_() noexcept {
  PetscFunctionBegin;
  PetscCall(PetscInfo(nullptr, "Finalizing memory map\n"));
  PetscCallCXX(map = map_type{});
  PetscFunctionReturn(0);
}

/*
  MemoryMap::search_for - retrieve an iterator to the key-value pair for a pointer in the map

  Input Parameter:
. ptr - pointer to search for

  Notes:
  Accounts for sub-regions, i.e. if ptr is contained within another pointers region, it returns
  the iterator to the super-pointers key-value pair.

  If ptr is not found, returns map.end()
*/
MemoryMap::map_type::const_iterator MemoryMap::search_for(const void *ptr) const noexcept {
  const auto end = map.end();
  const auto it  = map.find(const_cast<map_type::key_type>(ptr));

  // ptr was found, and points to an entire block
  if (it != end) return it;
  // wasn't found, but maybe its part of a block. have to search every block for it
  // clang-format off
  return std::find_if(map.begin(), end, [ptr](const map_type::const_iterator::value_type &map_it) {
    return map_it.second.contains(map_it.first, ptr);
  });
  // clang-format on
}

static MemoryMap memory_map;

// ==========================================================================================
// Utility functions
// ==========================================================================================

static PetscErrorCode PetscDeviceCheckCapable_Private(PetscDeviceContext dctx, bool cond, const char descr[]) {
  PetscFunctionBegin;
  PetscCheck(cond, PETSC_COMM_SELF, PETSC_ERR_SUP, "Device context (id: %" PetscInt64_FMT ", name: %s, type: %s) can only handle %s host memory", PetscObjectCast(dctx)->id, PetscObjectCast(dctx)->name, dctx->device ? PetscDeviceTypes[dctx->device->type] : "unknown", descr);
  PetscFunctionReturn(0);
}

// A helper utility, since register is called from PetscDeviceRegisterMemory() and
// PetscDevicAllocate(). The latter also needs the generated id, so instead of making it search
// the map again we just return it here
static PetscErrorCode PetscDeviceRegisterMemory_Private(const void *PETSC_RESTRICT ptr, PetscMemType mtype, std::size_t size, PetscObjectId *PETSC_RESTRICT id = nullptr) {
  auto      &map = memory_map.map;
  const auto it  = memory_map.search_for(ptr);

  PetscFunctionBegin;
  if (it == map.cend()) {
    // pointer was never registered with the map, insert it and bail
    const auto newid = PetscObjectNewId_Internal();

    if (PetscDefined(USE_DEBUG)) {
      const auto tmp = MemoryMap::PointerAttributes(mtype, newid, size);

      for (const auto &entry : map) {
        // REVIEW ME: maybe this should just be handled...
        PetscCheck(!tmp.contains(ptr, entry.first), PETSC_COMM_SELF, PETSC_ERR_ORDER, "Trying to register pointer %p (memtype %s, size %zu) but it appears you have already registered a sub-region of it (pointer %p, memtype %s, size %zu). Must register the larger region first", ptr, PetscMemTypeToString(mtype), size,
                   entry.first, PetscMemTypeToString(entry.second.mtype), entry.second.size);
      }
    }
    // clang-format off
    if (id) *id = newid;
    PetscCallCXX(map.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(const_cast<MemoryMap::map_type::key_type>(ptr)),
      std::forward_as_tuple(mtype, newid, size)
    ));
    // clang-format on
    PetscFunctionReturn(0);
  }
  if (PetscDefined(USE_DEBUG)) {
    const auto &old = it->second;

    PetscCheck(MemoryMap::PointerAttributes(mtype, old.id, size) == old, PETSC_COMM_SELF, PETSC_ERR_LIB, "Pointer %p appears to have been previously allocated with memtype %s, size %zu and assigned id %" PetscInt64_FMT ", which does not match new values: (mtype %s, size %zu, id %" PetscInt64_FMT ")", it->first,
               PetscMemTypeToString(old.mtype), old.size, old.id, PetscMemTypeToString(mtype), size, old.id);
  }
  if (id) *id = it->second.id;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceRegisterMemory - Register a pointer for use with device-aware memory system

  Not Collective

  Input Parameters:
+ ptr   - The pointer to register
. mtype - The `PetscMemType` of the pointer
- size  - The size (in bytes) of the memory region

  Notes:
  `ptr` need not point to the beginning of the memory range, however the user should register
  the

  It's OK to re-register the same `ptr` repeatedly (subsequent registrations do nothing)
  however the given `mtype` and `size` must match the original registration.

  `size` may be 0 (in which case this routine does nothing).

  Level: intermediate

.seealso: `PetscDeviceMalloc()`, `PetscDeviceArrayCopy()`, `PetscDeviceFree()`,
`PetscDeviceArrayZero()`
@*/
PetscErrorCode PetscDeviceRegisterMemory(const void *PETSC_RESTRICT ptr, PetscMemType mtype, std::size_t size) {
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscValidPointer(ptr, 1);
  if (PetscUnlikely(!size)) PetscFunctionReturn(0); // there is no point registering empty range
  PetscCall(PetscDeviceRegisterMemory_Private(ptr, mtype, size));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceAllocate - Allocate device-aware memory

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to allocate the memory
. clear - Whether or not the memory should be zeroed
. mtype - The type of memory to allocate
- n     - The amount (in bytes) to allocate

  Output Parameter:
. ptr - The pointer to store the result in

  Notes:
  The user should prefer `PetscDeviceMalloc()` over this routine as it automatically computes
  the size of the allocation based on the size of the datatype.

  Memory allocated with this function must be freed with `PetscDeviceFree()` or
  `PetscDeviceDeallocate()`.

  Note result stored `ptr` is immediately valid and the user may freely inspect or manipulate
  its value on function return, i.e.\:

.vb
  PetscInt *ptr;

  PetscDeviceAllocate(dctx, PETSC_FALSE, PETSC_MEMTYPE_DEVICE, 20, (void**)&ptr);

  PetscInt *sub_ptr = ptr + 10; // OK, no need to synchronize

  ptr[0] = 10; // ERROR, directly accessing contents of ptr is undefined until synchronization
.ve

  If `n` is zero, then `ptr` is set to `PETSC_NULLPTR`.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| -\- dctx -->
                         \- ptr ->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscGetMemType()`, `PetscDeviceFree()`,
`PetscDeviceDeallocate()`, `PetscDeviceArrayCopy()`, `PetscDeviceArrayZero()`
@*/
PetscErrorCode PetscDeviceAllocate(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, size_t n, void **PETSC_RESTRICT ptr) {
  PetscObjectId id = 0;

  PetscFunctionBegin;
  PetscValidPointer(ptr, 5);
  *ptr = nullptr;
  if (PetscUnlikely(!n)) PetscFunctionReturn(0);
  PetscCall(memory_map.register_finalize());
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  // get our pointer here
  if (dctx->ops->memalloc) {
    PetscUseTypeMethod(dctx, memalloc, clear, mtype, n, ptr);
  } else {
    PetscCall(PetscDeviceCheckCapable_Private(dctx, PetscMemTypeHost(mtype), "allocating"));
    PetscCall(PetscMallocA(1, clear, __LINE__, PETSC_FUNCTION_NAME, __FILE__, n, ptr));
  }
  PetscCall(PetscDeviceRegisterMemory_Private(*ptr, mtype, n, &id));
  // Note this is a "write" so that the next dctx to try and read from the pointer has to wait
  // for the allocation to be ready
  PetscCall(PetscDeviceContextMarkIntentFromID(dctx, id, PETSC_MEMORY_ACCESS_WRITE, "memory allocation"));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDeallocate - Free device-aware memory

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to free the memory
- ptr   - The pointer to free

  Notes:
  `ptr` must have been allocated using any of `PetscDeviceMalloc()`, `PetscDeviceCalloc()` or
  `PetscDeviceAllocate()`, or registered with the system via `PetscDeviceRegisterMemory()`.

  The user should prefer `PetscDeviceFree()` over this routine as it automatically sets `ptr`
  to `PETSC_NULLPTR` on successful deallocation.

  `ptr` may be `NULL`.

  DAG representation:
.vb
  time ->

  -> dctx -/- |= CALL =| - dctx ->
  -> ptr -/
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceAllocate()`
@*/
PetscErrorCode PetscDeviceDeallocate(PetscDeviceContext dctx, void *PETSC_RESTRICT ptr) {
  PetscFunctionBegin;
  if (ptr) {
    auto      &map      = memory_map.map;
    const auto found_it = map.find(const_cast<MemoryMap::map_type::key_type>(ptr));

    if (PetscUnlikelyDebug(found_it == map.end())) {
      // OK this is a bad pointer, now determine why
      const auto it = memory_map.search_for(ptr);

      // if it is map.cend() then no allocation owns it, meaning it was not allocated by us!
      PetscCheck(it != map.cend(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Pointer %p was not allocated via PetscDeviceAllocate()", ptr);
      // if we are here then we did allocate it but the user has tried to do something along
      // the lines of:
      //
      // allocate(&ptr, size);
      // deallocate(ptr+5);
      //
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempting to deallocate pointer %p which is a suballocation of %p (memtype %s, id %" PetscInt64_FMT ", size %zu bytes)", ptr, it->first, PetscMemTypeToString(it->second.mtype), it->second.id,
              it->second.size);
    }
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

    // mark intent BEFORE we free, note we mark as write so that we are made to wait on any
    // outstanding reads (don't want to kill the pointer before they are done)
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, found_it->second.id, PETSC_MEMORY_ACCESS_WRITE, "memory deallocation"));

    // do free
    if (dctx->ops->memfree) {
      PetscUseTypeMethod(dctx, memfree, found_it->second.mtype, (void **)&ptr);
    } else {
      PetscCall(PetscDeviceCheckCapable_Private(dctx, PetscMemTypeHost(found_it->second.mtype), "freeing"));
    }
    // if ptr still exists, then the device context could not handle it
    if (ptr) PetscCall(PetscFree(ptr));
    PetscCallCXX(map.erase(found_it));
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceMemcpy - Copy memory in a device-aware manner

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to copy the memory
. dest - The pointer to copy to
. src  - The pointer to copy from
- n    - The amount (in bytes) to copy

  Notes:
  Both `dest` and `src` must have been allocated by `PetscDeviceAllocate()` or
  `PetscDeviceMalloc()` or registered with `PetscDeviceRegisterMemory()`.

  `src` and `dest` cannot overlap.

  If both `src` and `dest` are on the host this routine is fully synchronous.

  The user should prefer `PetscDeviceArrayCopy()` over this routine as it automatically
  computes the number of bytes to copy from the size of the pointer types.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ->
  -> dest --------------------->
  -> src ---------------------->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceArrayCopy()`, `PetscDeviceMalloc()`, `PetscDeviceFree()`
@*/
PetscErrorCode PetscDeviceMemcpy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n) {
  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
  PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
  if (dest == src) PetscFunctionReturn(0);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  {
    const auto dest_it = memory_map.search_for(dest);
    const auto src_it  = memory_map.search_for(src);

    PetscAssert(dest_it != memory_map.map.cend(), PETSC_COMM_SELF, PETSC_ERR_POINTER, "Destination pointer %p was not registered with the memory tracker, call PetscDeviceRegisterMemory() on it", dest);
    PetscAssert(src_it != memory_map.map.cend(), PETSC_COMM_SELF, PETSC_ERR_POINTER, "Source pointer %p was not registered with the memory tracker, call PetscDeviceRegisterMemory() on it", src);

    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, src_it->second.id, PETSC_MEMORY_ACCESS_READ, "memory copy (src)"));
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, dest_it->second.id, PETSC_MEMORY_ACCESS_WRITE, "memory copy (dest)"));

    const auto mode = PetscMemTypeToDeviceCopyMode(dest_it->second.mtype, src_it->second.mtype);
    // perform the copy
    if (dctx->ops->memcopy) {
      PetscUseTypeMethod(dctx, memcopy, dest, src, n, mode);
    } else {
      // REVIEW ME: we might potentially need to sync here if the memory is device-allocated
      // (pinned) but being copied by a host dctx
      PetscCall(PetscDeviceCheckCapable_Private(dctx, mode == PETSC_DEVICE_COPY_HTOH, "copying"));
      PetscCall(PetscMemcpy(dest, src, n));
    }

    if (mode == PETSC_DEVICE_COPY_HTOD) {
      PetscCall(PetscLogCpuToGpu(n));
    } else if (mode == PETSC_DEVICE_COPY_DTOH) {
      PetscCall(PetscLogGpuToCpu(n));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceMemset - Memset device-aware memory

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to memset the memory
. ptr   - The pointer to the memory
. v     - The value to set
- n     - The amount (in bytes) to set

  Notes:
  `ptr` must have been allocated by `PetscDeviceAllocate()` or `PetscDeviceMalloc()` or
  registered with `PetscDeviceRegisterMemory()`.

  The user should prefer `PetscDeviceArrayZero()` over this routine as it automatically
  computes the number of bytes to copy from the size of the pointer types, though they should
  note that it only zeros memory.

  This routine is analogous to `memset()`. That is, this routine copies the value
  `static_cast<unsigned char>(v)` into each of the first count characters of the object pointed
  to by `dest`.

  If `dest` is on device, this routine is asynchronous.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ->
  -> dest --------------------->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceArrayZero()`, `PetscDeviceMalloc()`, `PetscDeviceFree()`
@*/
PetscErrorCode PetscDeviceMemset(PetscDeviceContext dctx, void *ptr, PetscInt v, std::size_t n) {
  PetscFunctionBegin;
  if (PetscUnlikely(!n)) PetscFunctionReturn(0);
  PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to memset a NULL pointer");
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  {
    const auto ptr_it = memory_map.search_for(ptr);

    // mark
    PetscAssert(ptr_it != memory_map.map.cend(), PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p was not registered with the memory tracker, call PetscDeviceRegisterMemory() on it", ptr);
    PetscCall(PetscDeviceContextMarkIntentFromID(dctx, ptr_it->second.id, PETSC_MEMORY_ACCESS_WRITE, "memory set"));

    // set
    if (dctx->ops->memset) {
      PetscUseTypeMethod(dctx, memset, ptr_it->second.mtype, ptr, v, n);
    } else {
      // REVIEW ME: we might potentially need to sync here if the memory is device-allocated
      // (pinned) but being memset by a host dctx
      PetscCall(PetscDeviceCheckCapable_Private(dctx, PetscMemTypeHost(ptr_it->second.mtype), "memsetting"));
      std::memset(ptr, static_cast<int>(v), n);
    }
  }
  PetscFunctionReturn(0);
}
