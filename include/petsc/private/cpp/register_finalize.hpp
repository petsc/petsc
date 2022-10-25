#ifndef PETSC_CPP_REGISTER_FINALIZE_HPP
#define PETSC_CPP_REGISTER_FINALIZE_HPP

#include <petscsys.h>

#if defined(__cplusplus)
  #include <petsc/private/cpp/crtp.hpp>

namespace
{

template <typename T>
PETSC_NODISCARD inline PetscErrorCode PetscCxxObjectRegisterFinalize(T *obj, MPI_Comm comm = PETSC_COMM_SELF) noexcept
{
  PetscContainer contain   = nullptr;
  const auto     finalizer = [](void *ptr) {
    PetscFunctionBegin;
    PetscCall(static_cast<T *>(ptr)->finalize());
    PetscFunctionReturn(0);
  };

  PetscFunctionBegin;
  PetscValidPointer(obj, 1);
  PetscCall(PetscContainerCreate(comm, &contain));
  PetscCall(PetscContainerSetPointer(contain, obj));
  PetscCall(PetscContainerSetUserDestroy(contain, std::move(finalizer)));
  PetscCall(PetscObjectRegisterDestroy(reinterpret_cast<PetscObject>(contain)));
  PetscFunctionReturn(0);
}

} // anonymous namespace

namespace Petsc
{

// ==========================================================================================
// RegisterFinalizeable
//
// A mixin class that enables registering a finalizer for a class instance to run during
// PetscFinalize(). Enables 3 public methods:
//
// 1. register_finalize() - Register the calling instance to run the member function
//    finalize_() during PetscFinalize(). It only registers the class once.
// 2. finalize() - Run the member function finalize_() immediately.
// 3. registered() - Query whether you are registered.
// ==========================================================================================
template <typename Derived>
class RegisterFinalizeable : public util::crtp<Derived, RegisterFinalizeable> {
public:
  using derived_type = Derived;

  PETSC_NODISCARD bool registered() const noexcept;
  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode finalize(Args &&...) noexcept;
  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode register_finalize(Args &&...) noexcept;

private:
  RegisterFinalizeable() = default;
  friend derived_type;

  // default implementations if the derived class does not want to implement them
  template <typename... Args>
  PETSC_NODISCARD static PetscErrorCode finalize_(Args &&...) noexcept;
  template <typename... Args>
  PETSC_NODISCARD static PetscErrorCode register_finalize_(Args &&...) noexcept;

  bool registered_ = false;
};

template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::finalize_(Args &&...) noexcept
{
  return 0;
}

template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::register_finalize_(Args &&...) noexcept
{
  return 0;
}

/*
  RegisterFinalizeable::registered - Determine if the class instance is registered

  Notes:
  Returns true if class is registered, false otherwise.
*/
template <typename D>
inline bool RegisterFinalizeable<D>::registered() const noexcept
{
  return registered_;
}

/*
  RegisterFinalizeable::finalize - Run the finalizer for a class

  Input Parameters:

. ...args - A set of arguments to pass to the finalizer

  Notes:
  It is not necessary to implement finalize_() in the derived class (though pretty much
  pointless), a default (no-op) implementation is provided.

  Runs the member function finalize_() with args forwarded.

  "Unregisters" the class from PetscFinalize(). However, it is safe for finalize_() to
  re-register itself (via register_finalize()). registered() is guaranteed to return false
  inside finalize_().
*/
template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::finalize(Args &&...args) noexcept
{
  PetscFunctionBegin;
  // order of setting registered_ to false matters here, if the finalizer wants to re-register
  // it should be able to
  if (this->underlying().registered()) {
    registered_ = false;
    PetscCall(this->underlying().finalize_(std::forward<Args>(args)...));
  }
  PetscFunctionReturn(0);
}

/*
  RegisterFinalizeable::register_finalize - Register a finalizer to run during PetscFinalize()

  Input Parameters:
. ...args - Additional arguments to pass to the register_finalize_() hook

  Notes:
  It is not necessary to implement register_finalize_() in the derived class. A default (no-op)
  implementation is provided.

  Before registering the class, the register_finalize_() hook function is run. This is useful
  for running any one-time setup code before registering. Subsequent invocations of this
  function (as long as registered() returns true) will not run register_finalize_() again.

  The class is considered registered before calling the hook, that is registered() will always
  return true inside register_finalize_(). register_finalize_() is allowed to immediately
  un-register the class (via finalize()). In this case the finalizer does not run at
  PetscFinalize(), and registered() returns false after this routine returns.
*/
template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::register_finalize(Args &&...args) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(this->underlying().registered())) PetscFunctionReturn(0);
  registered_ = true;
  PetscCall(this->underlying().register_finalize_(std::forward<Args>(args)...));
  // Check if registered before we commit to actually register-finalizing. register_finalize_()
  // is allowed to run its finalizer immediately
  if (this->underlying().registered()) PetscCall(PetscCxxObjectRegisterFinalize(this));
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_REGISTER_FINALIZE_HPP
