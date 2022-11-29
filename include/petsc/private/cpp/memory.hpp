#ifndef PETSC_CPP_MEMORY_HPP
#define PETSC_CPP_MEMORY_HPP

#include <petsc/private/petscimpl.h> // PetscValidPointer()
#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/type_traits.hpp> // remove_extent

#if defined(__cplusplus)
  #include <memory>
  #include <new>   // ::operator new(), ::operator delete()
  #include <stack> // ... take a wild guess

namespace Petsc
{

namespace util
{

  #if PETSC_CPP_VERSION >= 14
using std::make_unique;
  #else
namespace detail
{

// helpers shamelessly stolen from libcpp
template <class T>
struct unique_if {
  using unique_single = std::unique_ptr<T>;
};

template <class T>
struct unique_if<T[]> {
  using unique_array_unknown_bound = std::unique_ptr<T[]>;
};

template <class T, std::size_t N>
struct unique_if<T[N]> {
  using unique_array_unknown_bound = void;
};

} // namespace detail

template <class T, class... Args>
inline typename detail::unique_if<T>::unique_single make_unique(Args &&...args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
inline typename detail::unique_if<T>::unique_array_unknown_bound make_unique(std::size_t n)
{
  return std::unique_ptr<T>(new util::remove_extent_t<T>[n]());
}

template <class T, class... Args>
typename detail::unique_if<T>::unique_array_known_bound make_unique(Args &&...) = delete;
  #endif // PETSC_CPP_VERSION >= 14

} // namespace util

namespace memory
{

class PoolAllocator : public RegisterFinalizeable<PoolAllocator> {
public:
  using size_type      = std::size_t;
  using container_type = std::stack<void *>;

  explicit PoolAllocator(size_type = 0) noexcept;

  PETSC_NODISCARD size_type      block_size() const noexcept;
  PETSC_NODISCARD PetscErrorCode set_block_size(size_type) noexcept;
  PETSC_NODISCARD PetscErrorCode allocate(void **, size_type) noexcept;
  PETSC_NODISCARD PetscErrorCode deallocate(void **) noexcept;
  PETSC_NODISCARD PetscErrorCode clear() noexcept;

private:
  size_type      block_size_     = 0; // current block_size
  size_type      max_block_size_ = 0; // maximum block_size that block_size has ever been
  container_type stack_{};

  friend class RegisterFinalizeable<PoolAllocator>;
  // needed so that RegisterFinalizeable sees private finalize_()
  PETSC_NODISCARD PetscErrorCode finalize_() noexcept;
};

inline PoolAllocator::PoolAllocator(size_type block_size) noexcept : block_size_(block_size), max_block_size_(block_size) { }

inline typename PoolAllocator::size_type PoolAllocator::block_size() const noexcept
{
  return block_size_;
}

inline PetscErrorCode PoolAllocator::set_block_size(size_type block_size) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(block_size == block_size_)) PetscFunctionReturn(0);
  block_size_ = block_size;
  if (PetscUnlikely(block_size > max_block_size_)) {
    // new block_size is greater than our max, so we must discard our stack
    max_block_size_ = block_size;
    PetscCall(clear());
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode PoolAllocator::allocate(void **ptr, size_type size) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(ptr, 1);
  *ptr = nullptr;
  PetscCall(this->register_finalize());
  PetscCall(set_block_size(size));
  if (stack_.empty()) {
    PetscCallCXX(*ptr = ::new char[block_size()]);
  } else {
    *ptr = stack_.top();
    PetscCallCXX(stack_.pop());
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode PoolAllocator::deallocate(void **ptr) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(ptr, 1);
  if (!*ptr) PetscFunctionReturn(0);
  if (PetscLikely(this->registered())) {
    PetscCallCXX(stack_.push(*ptr));
  } else {
    // this is necessary if an object is "reclaimed" within another PetscFinalize() registered
    // cleanup after this object pool has returned from it's finalizer. In this case, instead
    // of pushing onto the stack we just delete the pointer directly
    ::delete[] static_cast<char *>(*ptr);
  }
  *ptr = nullptr;
  PetscFunctionReturn(0);
}

inline PetscErrorCode PoolAllocator::clear() noexcept
{
  PetscFunctionBegin;
  while (!stack_.empty()) {
    ::delete[] static_cast<char *>(stack_.top());
    PetscCallCXX(stack_.pop());
  }
  PetscFunctionReturn(0);
}

inline PetscErrorCode PoolAllocator::finalize_() noexcept
{
  PetscFunctionBegin;
  PetscCall(clear());
  PetscCallCXX(stack_ = container_type{});
  PetscFunctionReturn(0);
}

template <typename Derived>
class PoolAllocated {
public:
  using allocator_type = PoolAllocator;
  using size_type      = typename allocator_type::size_type;

  PETSC_NODISCARD static void *operator new(size_type) noexcept;
  static void                  operator delete(void *) noexcept;

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
  PetscCallAbort(PETSC_COMM_SELF, pool().allocate(&ptr, size));
  PetscFunctionReturn(ptr);
}

template <typename D>
inline void PoolAllocated<D>::operator delete(void *ptr) noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, pool().deallocate(&ptr));
  PetscFunctionReturnVoid();
}

template <typename D>
inline typename PoolAllocated<D>::allocator_type &PoolAllocated<D>::pool() noexcept
{
  return pool_;
}

} // namespace memory

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_MEMORY_HPP
