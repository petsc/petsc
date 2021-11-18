#ifndef PETSCOBJECTPOOL_HPP
#define PETSCOBJECTPOOL_HPP

#include <petscsys.h>
#if !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#error "ObjectPool requires c++11"
#endif

#if defined(__cplusplus)

#include <stack>
#include <type_traits>

namespace Petsc
{

// Allocator ABC for interoperability with C ctors and dtors.
template <typename T>
class Allocator
{
public:
  using value_type = T;

  PETSC_NODISCARD PetscErrorCode create(value_type*)  PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode destroy(value_type&) PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode reset(value_type&)   PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode finalize()           PETSC_NOEXCEPT;

protected:
  // make the constructor protected, this forces this class to be derived from to ever be
  // instantiated
  Allocator() = default;
};

// Default allocator that performs the bare minimum of petsc object creation and
// desctruction
template <typename T>
class CAllocator : public Allocator<T>
{
public:
  using allocator_type = Allocator<T>;
  using value_type     = typename allocator_type::value_type;

  PETSC_NODISCARD PetscErrorCode create(value_type *obj) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscNew(obj);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode destroy(value_type &obj) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = (*obj->ops->destroy)(obj);CHKERRQ(ierr);
    ierr = PetscHeaderDestroy(&obj);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode reset(value_type &obj) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = this->destroy(obj);CHKERRQ(ierr);
    ierr = this->create(&obj);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode finalize() const PETSC_NOEXCEPT { return 0; }
};

// Base class to object pool, defines helpful typedefs and stores the allocator instance
template <typename T, class _Allocator>
class ObjectPoolBase
{
public:
  using allocator_type = _Allocator;
  using value_type     = typename allocator_type::value_type;

protected:
  allocator_type _alloc;

  PETSC_NODISCARD allocator_type& __getAllocator() PETSC_NOEXCEPT { return _alloc; }

  PETSC_NODISCARD const allocator_type& __getAllocator() const PETSC_NOEXCEPT { return _alloc; }

  // default constructor
  constexpr ObjectPoolBase() PETSC_NOEXCEPT(std::is_nothrow_default_constructible<allocator_type>::value)
    : _alloc()
  { }

  // const copy constructor
  explicit ObjectPoolBase(const allocator_type &alloc) : _alloc(alloc) { }

  // move constructor
  explicit ObjectPoolBase(allocator_type &&alloc) PETSC_NOEXCEPT(std::is_nothrow_move_assignable<allocator_type>::value)
    : _alloc(std::move(alloc))
  { }

  static_assert(std::is_base_of<Allocator<value_type>,_Allocator>::value,"Allocator type must be subclass of Petsc::Allocator");
};

// default implementation, use the petsc c allocator
template <typename T, class _Allocator = CAllocator<T>> class ObjectPool;

// multi-purpose basic object-pool, useful for recirculating old "destroyed" objects. Uses
// a stack to take advantage of LIFO for memory locallity. Registers all objects to be
// cleaned up on PetscFinalize()
template <typename T, class _Allocator>
class ObjectPool : ObjectPoolBase<T,_Allocator>
{
protected:
  using base_type = ObjectPoolBase<T,_Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
  using value_type     = typename base_type::value_type;
  using stack_type     = std::stack<value_type>;

private:
  stack_type _stack;
  bool       _registered = false;

  PETSC_NODISCARD PetscErrorCode __finalizer() PETSC_NOEXCEPT;
  PETSC_NODISCARD static PetscErrorCode __staticFinalizer(void*) PETSC_NOEXCEPT;
  PETSC_NODISCARD PetscErrorCode __registerFinalize() PETSC_NOEXCEPT;

public:
  // default constructor
  constexpr ObjectPool() PETSC_NOEXCEPT(std::is_nothrow_default_constructible<allocator_type>::value)
    : _stack()
  { }

  // destructor
  ~ObjectPool() PETSC_NOEXCEPT
  {
    PetscErrorCode ierr = __finalizer();CHKERRABORT(PETSC_COMM_SELF,ierr);
  }

  // copy constructor
  ObjectPool(ObjectPool &other) PETSC_NOEXCEPT(std::is_nothrow_copy_constructible<stack_type>::value)
    : _stack(other._stack),_registered(other._registered)
  { }

  // const copy constructor
  ObjectPool(const ObjectPool &other) PETSC_NOEXCEPT(std::is_nothrow_copy_constructible<stack_type>::value)
    : _stack(other._stack),_registered(other._registered)
  { }

  // move constructor
  ObjectPool(ObjectPool &&other) PETSC_NOEXCEPT(std::is_nothrow_move_constructible<stack_type>::value)
    : _stack(std::move(other._stack)),_registered(std::move(other._registered))
  { }

  // copy constructor with allocator
  explicit ObjectPool(const allocator_type &alloc) : base_type(alloc) { }

  // move constructor with allocator
  explicit ObjectPool(allocator_type &&alloc) PETSC_NOEXCEPT(std::is_nothrow_move_constructible<allocator_type>::value)
    : base_type(std::move(alloc))
  { }

  // Retrieve an object from the pool, if the pool is empty a new object is created instead
  PETSC_NODISCARD PetscErrorCode get(value_type&)      PETSC_NOEXCEPT;
  // Return an object to the pool, the object need not necessarily have been created by
  // the pool, note this only accepts r-value references. The pool takes ownership of all
  // managed objects.
  PETSC_NODISCARD PetscErrorCode reclaim(value_type&&) PETSC_NOEXCEPT;

  // operators
  template <typename T_, class A_>
  PetscBool friend operator==(const ObjectPool<T_,A_>&,const ObjectPool<T_,A_>&);

  template <typename T_, class A_>
  PetscBool friend operator< (const ObjectPool<T_,A_>&,const ObjectPool<T_,A_>&);
};

template <typename T, class _Allocator>
inline PetscBool operator==(const ObjectPool<T,_Allocator> &l,const ObjectPool<T,_Allocator> &r)
{
  return static_cast<PetscBool>(l._stack == r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator< (const ObjectPool<T,_Allocator> &l, const ObjectPool<T,_Allocator> &r)
{
  return static_cast<PetscBool>(l._stack < r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator!=(const ObjectPool<T,_Allocator> &l, const ObjectPool<T,_Allocator> &r)
{
  return !(l._stack == r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator> (const ObjectPool<T,_Allocator> &l, const ObjectPool<T,_Allocator> &r)
{
  return r._stack < l._stack;
}

template <typename T, class _Allocator>
inline PetscBool operator>=(const ObjectPool<T,_Allocator> &l, const ObjectPool<T,_Allocator> &r)
{
  return !(l._stack < r._stack);
}

template <typename T, class _Allocator>
inline PetscBool operator<=(const ObjectPool<T,_Allocator> &l, const ObjectPool<T,_Allocator> &r)
{
  return !(r._stack < l._stack);
}

template <typename T, class _Allocator>
inline PetscErrorCode ObjectPool<T,_Allocator>::__finalizer() PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (!_stack.empty()) {
    // we do CHKERRQ __after__ the CHKERCXX on the off chance that someone uses the CXX
    // error handler, we don't want to catch our own exception!
    CHKERRCXX(ierr = base_type::__getAllocator().destroy(_stack.top()));CHKERRQ(ierr);
    CHKERRCXX(_stack.pop());
  }
  ierr = base_type::__getAllocator().finalize();CHKERRQ(ierr);
  _registered = false;
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
PetscErrorCode ObjectPool<T,_Allocator>::__staticFinalizer(void *obj) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = static_cast<ObjectPool<T,_Allocator>*>(obj)->__finalizer();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
inline PetscErrorCode ObjectPool<T,_Allocator>::__registerFinalize() PETSC_NOEXCEPT
{
  PetscErrorCode ierr;
  PetscContainer contain;

  PetscFunctionBegin;
  if (PetscLikely(_registered)) PetscFunctionReturn(0);
  /* use a PetscContainer as a form of thunk, it holds not only a pointer to this but
     also the pointer to the static member function, which just converts the thunk back
     to this. none of this would be needed if PetscRegisterFinalize() just took a void*
     itself though...  */
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&contain);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(contain,this);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(contain,__staticFinalizer);CHKERRQ(ierr);
  ierr = PetscObjectRegisterDestroy(reinterpret_cast<PetscObject>(contain));CHKERRQ(ierr);
  _registered = true;
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
inline PetscErrorCode ObjectPool<T,_Allocator>::get(value_type &obj) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = __registerFinalize();CHKERRQ(ierr);
  if (_stack.empty()) {
    ierr = base_type::__getAllocator().create(&obj);CHKERRQ(ierr);
  } else {
    CHKERRCXX(obj = std::move(_stack.top()));
    CHKERRCXX(_stack.pop());
  }
  PetscFunctionReturn(0);
}

template <typename T, class _Allocator>
inline PetscErrorCode ObjectPool<T,_Allocator>::reclaim(value_type &&obj) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLikely(_registered)) {
    // allows const allocator_t& to be used if allocator defines a const reset
    ierr = base_type::__getAllocator().reset(obj);CHKERRQ(ierr);
    CHKERRCXX(_stack.push(std::move(obj)));
  } else {
    // this is necessary if an object is "reclaimed" within another PetscFinalize() registered
    // cleanup after this object pool has returned from it's finalizer. In this case, instead
    // of pushing onto the stack we just destroy the object directly
    ierr = base_type::__getAllocator().destroy(obj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCOBJECTPOOL_HPP */
