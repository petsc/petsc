#ifndef PETSCOBJECTPOOL_HPP
#define PETSCOBJECTPOOL_HPP

#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/memory.hpp>
#include <petscsys.h>

#if defined(__cplusplus)
  #include <stack>

namespace Petsc
{

// Allocator ABC for interoperability with C ctors and dtors.
template <typename T>
class AllocatorBase {
public:
  using value_type = T;

  PETSC_NODISCARD PetscErrorCode        create(value_type *) noexcept  = delete;
  PETSC_NODISCARD PetscErrorCode        destroy(value_type &) noexcept = delete;
  PETSC_NODISCARD static PetscErrorCode reset(value_type &) noexcept { return 0; }
  PETSC_NODISCARD static PetscErrorCode finalize() noexcept { return 0; }

protected:
  // make the constructor protected, this forces this class to be derived from to ever be
  // instantiated
  constexpr AllocatorBase() noexcept = default;
};

// REVIEW ME:
// TODO: The object pool should use the generic pool allocator, as the object pool is nothing
// more than a pool allocator with fixed block_size!
namespace detail
{

// Base class to object pool, defines helpful typedefs and stores the allocator instance
template <class Allocator>
class ObjectPoolBase {
public:
  using allocator_type = Allocator;
  using value_type     = typename allocator_type::value_type;
  static_assert(std::is_base_of<AllocatorBase<value_type>, Allocator>::value, "");

  PETSC_NODISCARD const allocator_type &callocator() const noexcept { return alloc_; }
  PETSC_NODISCARD const allocator_type &allocator() const noexcept { return callocator(); }
  PETSC_NODISCARD allocator_type       &allocator() noexcept { return alloc_; }

protected:
  allocator_type alloc_;
};

} // namespace detail

template <typename T, typename Allocator, typename Container = typename std::stack<T>::container_type>
class ObjectPool;

// multi-purpose basic object-pool, useful for recirculating old "destroyed" objects. Uses
// a stack to take advantage of LIFO for memory locality. Registers all objects to be
// cleaned up on PetscFinalize()
template <typename T, typename Allocator, typename Container>
class ObjectPool : detail::ObjectPoolBase<Allocator>, public RegisterFinalizeable<ObjectPool<T, Allocator, Container>> {
protected:
  using base_type = detail::ObjectPoolBase<Allocator>;

public:
  using value_type     = T;
  using container_type = Container;
  using stack_type     = std::stack<value_type, container_type>;
  using typename base_type::allocator_type;
  static_assert(std::is_same<value_type, typename base_type::value_type>::value, "");

  // destructor
  ~ObjectPool() noexcept(std::is_nothrow_destructible<base_type>::value &&std::is_nothrow_destructible<stack_type>::value)
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, this->finalize());
    PetscFunctionReturnVoid();
  }

  // Retrieve an object from the pool, if the pool is empty a new object is created instead
  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode allocate(value_type *, Args &&...) noexcept;
  // Return an object to the pool, the object need not necessarily have been created by
  // the pool. Note the rvalue reference, the object pool takes immediate ownership of the
  // object
  PETSC_NODISCARD PetscErrorCode deallocate(value_type &&) noexcept;

private:
  // so RegisterFinalizeable sees finalize_()
  friend class RegisterFinalizeable<ObjectPool<T, Allocator, Container>>;
  stack_type stack_{};

  PETSC_NODISCARD PetscErrorCode finalize_() noexcept;
};

template <typename T, typename Allocator, typename Container>
inline PetscErrorCode ObjectPool<T, Allocator, Container>::finalize_() noexcept
{
  PetscFunctionBegin;
  while (!stack_.empty()) {
    PetscCall(this->allocator().destroy(std::move(stack_.top())));
    stack_.pop();
  }
  PetscCall(this->allocator().finalize());
  PetscFunctionReturn(0);
}

template <typename T, typename Allocator, typename Container>
template <typename... Args>
inline PetscErrorCode ObjectPool<T, Allocator, Container>::allocate(value_type *obj, Args &&...args) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(obj, 1);
  PetscCall(this->register_finalize());
  if (stack_.empty()) {
    PetscCall(this->allocator().create(obj, std::forward<Args>(args)...));
    PetscFunctionReturn(0);
  }
  PetscCall(this->allocator().reset(stack_.top(), std::forward<Args>(args)...));
  *obj = std::move(stack_.top());
  stack_.pop();
  PetscFunctionReturn(0);
}

template <typename T, typename Allocator, typename Container>
inline PetscErrorCode ObjectPool<T, Allocator, Container>::deallocate(value_type &&obj) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(this->registered())) {
    // allows const allocator_t& to be used if allocator defines a const reset
    PetscCallCXX(stack_.push(std::move(obj)));
  } else {
    // this is necessary if an object is "reclaimed" within another PetscFinalize() registered
    // cleanup after this object pool has returned from it's finalizer. In this case, instead
    // of pushing onto the stack we just destroy the object directly
    PetscCall(this->allocator().destroy(std::move(obj)));
  }
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCOBJECTPOOL_HPP */
