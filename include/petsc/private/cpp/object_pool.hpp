#pragma once

#include <petsc/private/petscimpl.h> // PetscAssertPointer()
#include <petsc/private/mempoison.h> // PetscPoison/UnpoisonMemoryRegion()

#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/utility.hpp> // util::exchange(), std::pair
#include <petsc/private/cpp/unordered_map.hpp>
#include <petsc/private/cpp/memory.hpp> // std::align(), std::unique_ptr

#include <cstddef> // std::size_t
#include <vector>  // std::take_a_wild_guess

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
  using pool_type  = std::vector<std::pair<align_type, UnorderedMap<size_type, std::vector<void *>>>>;

  PoolAllocator() noexcept                            = default;
  PoolAllocator(PoolAllocator &&) noexcept            = default;
  PoolAllocator &operator=(PoolAllocator &&) noexcept = default;

  // the pool carries raw memory and is not copyable
  PoolAllocator(const PoolAllocator &)            = delete;
  PoolAllocator &operator=(const PoolAllocator &) = delete;

  ~PoolAllocator() noexcept;

  PetscErrorCode try_allocate(void **, size_type, align_type, bool *) noexcept;
  PetscErrorCode allocate(void **, size_type, align_type, bool * = nullptr) noexcept;
  PetscErrorCode deallocate(void **, size_type, align_type) noexcept;

  static PetscErrorCode get_attributes(const void *, size_type *, align_type *) noexcept;
  static PetscErrorCode unpoison(const void *, size_type *) noexcept;
  static PetscErrorCode repoison(const void *, size_type) noexcept;

  template <typename T>
  PetscErrorCode for_each(T &&) noexcept;

  class LockGuard {
  public:
    LockGuard() = delete;

  private:
    friend class PoolAllocator;

    explicit LockGuard(PoolAllocator *pool) noexcept : pool_{pool} { ++pool_->locked_; }

    struct PoolUnlocker {
      void operator()(PoolAllocator *pool) const noexcept { --pool->locked_; }
    };

    std::unique_ptr<PoolAllocator, PoolUnlocker> pool_{};
  };

  LockGuard lock_guard() noexcept;

private:
  pool_type pool_;
  int       locked_ = 0;

  struct AllocationHeader;

  PETSC_NODISCARD static constexpr size_type total_size_(size_type, align_type) noexcept;

  static PetscErrorCode valid_alignment_(align_type) noexcept;
  static PetscErrorCode extract_header_(void *, AllocationHeader **, bool = true) noexcept;
  static PetscErrorCode allocate_ptr_(size_type, align_type, void **) noexcept;

  static PetscErrorCode delete_ptr_(void **) noexcept;

  PETSC_NODISCARD pool_type       &pool() noexcept { return pool_; }
  PETSC_NODISCARD const pool_type &pool() const noexcept { return pool_; }

  PETSC_NODISCARD typename pool_type::iterator       find_align_(align_type) noexcept;
  PETSC_NODISCARD typename pool_type::const_iterator find_align_(align_type) const noexcept;

public:
  PetscErrorCode clear_(size_type * = nullptr) noexcept;
  PetscErrorCode finalize_() noexcept;
};

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
    PetscCall(this->underlying().construct_(ptr, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode destroy(value_type *ptr) const noexcept
  {
    const Derived &underlying = this->underlying();

    PetscFunctionBegin;
    PetscCall(underlying.destroy_(ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename... Args>
  PetscErrorCode reset(value_type *val, Args &&...args) const noexcept
  {
    const Derived &underlying = this->underlying();

    PetscFunctionBegin;
    PetscCall(underlying.reset_(val, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode invalidate(value_type *ptr) const noexcept
  {
    const Derived &underlying = this->underlying();

    PetscFunctionBegin;
    PetscCall(underlying.invalidate_(ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

protected:
  template <typename... Args>
  static PetscErrorCode construct_(value_type *ptr, Args &&...args) noexcept
  {
    PetscFunctionBegin;
    PetscAssertPointer(ptr, 1);
    PetscCallCXX(util::construct_at(ptr, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode destroy_(value_type *ptr) noexcept
  {
    PetscFunctionBegin;
    if (ptr) {
      PetscAssertPointer(ptr, 1);
      PetscCallCXX(util::destroy_at(ptr));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename... Args>
  PetscErrorCode reset_(value_type *val, Args &&...args) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(this->underlying().construct(val, std::forward<Args>(args)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode invalidate_(value_type *ptr) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(this->underlying().destroy(ptr));
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
  {
    auto _ = this->allocator().lock_guard();

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
  }
  PetscCall(this->allocator().clear_());
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
  PetscAssertPointer(obj, 1);
  // order is deliberate! We register our finalizer before the pool does so since we need to
  // destroy the objects within it before it deletes their memory.
  PetscCall(this->allocator().allocate(&mem, sizeof(value_type), static_cast<align_type>(alignof(value_type)), &allocated_from_pool));
  PetscCall(this->register_finalize());
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
  PetscAssertPointer(obj, 1);
  PetscCall(this->register_finalize());
  PetscCall(this->constructor().invalidate(*obj));
  PetscCall(this->allocator().deallocate(reinterpret_cast<void **>(obj), sizeof(value_type), static_cast<align_type>(alignof(value_type))));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace Petsc
