#ifndef PETSCOBJECTPOOL_HPP
#define PETSCOBJECTPOOL_HPP

#include <petscsys.h>

#if defined(__cplusplus)

#include <stack>
#include <type_traits>

namespace Petsc
{

// Allocator ABC for interoperability with C ctors and dtors.
template <typename T>
class AllocatorBase
{
public:
  using value_type = T;

  PETSC_NODISCARD PetscErrorCode create(value_type*)  noexcept;
  PETSC_NODISCARD PetscErrorCode destroy(value_type&) noexcept;
  PETSC_NODISCARD PetscErrorCode reset(value_type&)   noexcept;
  PETSC_NODISCARD PetscErrorCode finalize()           noexcept;

protected:
  // make the constructor protected, this forces this class to be derived from to ever be
  // instantiated
  AllocatorBase() noexcept = default;
};

// Default allocator that performs the bare minimum of petsc object creation and
// desctruction
template <typename T>
class CAllocator : public AllocatorBase<T>
{
public:
  using allocator_type = AllocatorBase<T>;
  using value_type     = typename allocator_type::value_type;

  PETSC_NODISCARD PetscErrorCode create(value_type *obj) const noexcept
  {

    PetscFunctionBegin;
    CHKERRQ(PetscNew(obj));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode destroy(value_type &obj) const noexcept
  {

    PetscFunctionBegin;
    CHKERRQ((*obj->ops->destroy)(obj));
    CHKERRQ(PetscHeaderDestroy(&obj));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode reset(value_type &obj) const noexcept
  {

    PetscFunctionBegin;
    CHKERRQ(this->destroy(obj));
    CHKERRQ(this->create(&obj));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode finalize() const noexcept { return 0; }
};

namespace detail
{

// Base class to object pool, defines helpful typedefs and stores the allocator instance
template <typename T, class Allocator>
class ObjectPoolBase
{
public:
  using allocator_type = Allocator;
  using value_type     = typename allocator_type::value_type;

protected:
  allocator_type alloc_;

  PETSC_NODISCARD       allocator_type&  allocator()       noexcept { return alloc_; }
  PETSC_NODISCARD const allocator_type& callocator() const noexcept { return alloc_; }

  // default constructor
  constexpr ObjectPoolBase() noexcept(std::is_nothrow_default_constructible<allocator_type>::value)
    : alloc_()
  { }

  // const copy constructor
  explicit ObjectPoolBase(const allocator_type &alloc) : alloc_(alloc) { }

  // move constructor
  explicit ObjectPoolBase(allocator_type &&alloc)
    noexcept(std::is_nothrow_move_assignable<allocator_type>::value)
    : alloc_(std::move(alloc))
  { }

  static_assert(std::is_base_of<AllocatorBase<value_type>,Allocator>::value,"");
};

} // namespace detail

// default implementation, use the petsc c allocator
template <typename T, class Allocator = CAllocator<T>> class ObjectPool;

// multi-purpose basic object-pool, useful for recirculating old "destroyed" objects. Uses
// a stack to take advantage of LIFO for memory locallity. Registers all objects to be
// cleaned up on PetscFinalize()
template <typename T, class Allocator>
class ObjectPool : detail::ObjectPoolBase<T,Allocator>
{
protected:
  using base_type = detail::ObjectPoolBase<T,Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
  using value_type     = typename base_type::value_type;
  using stack_type     = std::stack<value_type>;
  using base_type::allocator;
  using base_type::callocator;

private:
  stack_type stack_;
  bool       registered_ = false;

  PETSC_NODISCARD        PetscErrorCode registerFinalize_()     noexcept;
  PETSC_NODISCARD        PetscErrorCode finalizer_()            noexcept;
  PETSC_NODISCARD static PetscErrorCode staticFinalizer_(void*) noexcept;

public:
  // default constructor
  constexpr ObjectPool() noexcept(std::is_nothrow_default_constructible<allocator_type>::value)
    : stack_()
  { }

  // destructor
  ~ObjectPool() noexcept
  {
    CHKERRABORT(PETSC_COMM_SELF,finalizer_());
  }

  // copy constructor
  ObjectPool(ObjectPool &other) noexcept(std::is_nothrow_copy_constructible<stack_type>::value)
    : stack_(other.stack_),registered_(other.registered_)
  { }

  // const copy constructor
  ObjectPool(const ObjectPool &other)
    noexcept(std::is_nothrow_copy_constructible<stack_type>::value)
    : stack_(other.stack_),registered_(other.registered_)
  { }

  // move constructor
  ObjectPool(ObjectPool &&other) noexcept(std::is_nothrow_move_constructible<stack_type>::value)
    : stack_(std::move(other.stack_)),registered_(std::move(other.registered_))
  { }

  // copy constructor with allocator
  explicit ObjectPool(const allocator_type &alloc) : base_type(alloc) { }

  // move constructor with allocator
  explicit ObjectPool(allocator_type &&alloc)
    noexcept(std::is_nothrow_move_constructible<allocator_type>::value)
    : base_type(std::move(alloc))
  { }

  // Retrieve an object from the pool, if the pool is empty a new object is created instead
  PETSC_NODISCARD PetscErrorCode get(value_type&)      noexcept;

  // Return an object to the pool, the object need not necessarily have been created by
  // the pool, note this only accepts r-value references. The pool takes ownership of all
  // managed objects.
  PETSC_NODISCARD PetscErrorCode reclaim(value_type&&) noexcept;

  // operators
  template <typename T_, class A_>
  PetscBool friend operator==(const ObjectPool<T_,A_>&,const ObjectPool<T_,A_>&) noexcept;

  template <typename T_, class A_>
  PetscBool friend operator< (const ObjectPool<T_,A_>&,const ObjectPool<T_,A_>&) noexcept;
};

template <typename T, class Allocator>
inline PetscBool operator==(const ObjectPool<T,Allocator> &l,const ObjectPool<T,Allocator> &r) noexcept
{
  return static_cast<PetscBool>(l.stack_ == r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator< (const ObjectPool<T,Allocator> &l, const ObjectPool<T,Allocator> &r) noexcept
{
  return static_cast<PetscBool>(l.stack_ < r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator!=(const ObjectPool<T,Allocator> &l, const ObjectPool<T,Allocator> &r) noexcept
{
  return !(l.stack_ == r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator> (const ObjectPool<T,Allocator> &l, const ObjectPool<T,Allocator> &r) noexcept
{
  return r.stack_ < l.stack_;
}

template <typename T, class Allocator>
inline PetscBool operator>=(const ObjectPool<T,Allocator> &l, const ObjectPool<T,Allocator> &r) noexcept
{
  return !(l.stack_ < r.stack_);
}

template <typename T, class Allocator>
inline PetscBool operator<=(const ObjectPool<T,Allocator> &l, const ObjectPool<T,Allocator> &r) noexcept
{
  return !(r.stack_ < l.stack_);
}

template <typename T, class Allocator>
inline PetscErrorCode ObjectPool<T,Allocator>::finalizer_() noexcept
{
  PetscFunctionBegin;
  while (!stack_.empty()) {
    // we do CHKERRQ __after__ the CHKERCXX on the off chance that someone uses the CXX
    // error handler, we don't want to catch our own exception!
    CHKERRCXX(CHKERRQ(this->allocator().destroy(stack_.top())));
    CHKERRCXX(stack_.pop());
  }
  CHKERRQ(this->allocator().finalize());
  registered_ = false;
  PetscFunctionReturn(0);
}

template <typename T, class Allocator>
inline PetscErrorCode ObjectPool<T,Allocator>::staticFinalizer_(void *obj) noexcept
{
  PetscFunctionBegin;
  CHKERRQ(static_cast<ObjectPool<T,Allocator>*>(obj)->finalizer_());
  PetscFunctionReturn(0);
}

template <typename T, class Allocator>
inline PetscErrorCode ObjectPool<T,Allocator>::registerFinalize_() noexcept
{
  PetscContainer contain;

  PetscFunctionBegin;
  if (PetscLikely(registered_)) PetscFunctionReturn(0);
  /* use a PetscContainer as a form of thunk, it holds not only a pointer to this but
     also the pointer to the static member function, which just converts the thunk back
     to this. none of this would be needed if PetscRegisterFinalize() just took a void*
     itself though...  */
  CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF,&contain));
  CHKERRQ(PetscContainerSetPointer(contain,this));
  CHKERRQ(PetscContainerSetUserDestroy(contain,staticFinalizer_));
  CHKERRQ(PetscObjectRegisterDestroy(reinterpret_cast<PetscObject>(contain)));
  registered_ = true;
  PetscFunctionReturn(0);
}

template <typename T, class Allocator>
inline PetscErrorCode ObjectPool<T,Allocator>::get(value_type &obj) noexcept
{
  PetscFunctionBegin;
  CHKERRQ(registerFinalize_());
  if (stack_.empty()) {
    CHKERRQ(this->allocator().create(&obj));
  } else {
    CHKERRCXX(obj = std::move(stack_.top()));
    CHKERRCXX(stack_.pop());
  }
  PetscFunctionReturn(0);
}

template <typename T, class Allocator>
inline PetscErrorCode ObjectPool<T,Allocator>::reclaim(value_type &&obj) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(registered_)) {
    // allows const allocator_t& to be used if allocator defines a const reset
    CHKERRQ(this->allocator().reset(obj));
    CHKERRCXX(stack_.push(std::move(obj)));
  } else {
    // this is necessary if an object is "reclaimed" within another PetscFinalize() registered
    // cleanup after this object pool has returned from it's finalizer. In this case, instead
    // of pushing onto the stack we just destroy the object directly
    CHKERRQ(this->allocator().destroy(std::move(obj)));
  }
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCOBJECTPOOL_HPP */
