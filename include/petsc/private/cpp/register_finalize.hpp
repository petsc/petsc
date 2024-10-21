#pragma once

#include <petscsys.h>

#include <petsc/private/cpp/crtp.hpp>
#include <petsc/private/cpp/type_traits.hpp>

template <typename T>
inline PetscErrorCode PetscCxxObjectRegisterFinalize(T *obj, MPI_Comm comm = PETSC_COMM_SELF) noexcept
{
  PetscContainer contain   = nullptr;
  const auto     finalizer = [](void **ptr) {
    PetscFunctionBegin;
    PetscCall(static_cast<T *>(*ptr)->finalize());
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  PetscFunctionBegin;
  PetscAssertPointer(obj, 1);
  PetscCall(PetscContainerCreate(comm, &contain));
  PetscCall(PetscContainerSetPointer(contain, obj));
  PetscCall(PetscContainerSetCtxDestroy(contain, std::move(finalizer)));
  PetscCall(PetscObjectRegisterDestroy(reinterpret_cast<PetscObject>(contain)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
class RegisterFinalizeable : public util::crtp<RegisterFinalizeable, Derived> {
public:
  using derived_type = Derived;

  PETSC_NODISCARD bool registered() const noexcept;
  template <typename... Args>
  PetscErrorCode finalize(Args &&...) noexcept;
  template <typename... Args>
  PetscErrorCode finalize(Args &&...) const noexcept;
  template <typename... Args>
  PetscErrorCode register_finalize(Args &&...) noexcept;
  template <typename... Args>
  PetscErrorCode register_finalize(Args &&...) const noexcept;

private:
  constexpr RegisterFinalizeable() noexcept = default;
  friend derived_type;

  template <typename Self, typename... Args>
  static PetscErrorCode do_finalize_(Self &&, Args &&...) noexcept;
  template <typename Self, typename... Args>
  static PetscErrorCode do_register_finalize_(Self &&, Args &&...) noexcept;

  // default implementations if the derived class does not want to implement them
  template <typename... Args>
  static constexpr PetscErrorCode finalize_(Args &&...) noexcept;
  template <typename... Args>
  static constexpr PetscErrorCode register_finalize_(Args &&...) noexcept;

  mutable bool registered_ = false;
};

template <typename D>
template <typename Self, typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::do_finalize_(Self &&self, Args &&...args) noexcept
{
  PetscFunctionBegin;
  // order of setting registered_ to false matters here, if the finalizer wants to re-register
  // it should be able to
  if (self.underlying().registered()) {
    self.registered_ = false;
    PetscCall(self.underlying().finalize_(std::forward<Args>(args)...));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename Self, typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::do_register_finalize_(Self &&self, Args &&...args) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(self.underlying().registered())) PetscFunctionReturn(PETSC_SUCCESS);
  self.registered_ = true;
  PetscCall(self.underlying().register_finalize_(std::forward<Args>(args)...));
  // Check if registered before we commit to actually register-finalizing. register_finalize_()
  // is allowed to run its finalizer immediately
  if (self.underlying().registered()) {
    using decayed_type = util::decay_t<Self>;

    PetscCall(PetscCxxObjectRegisterFinalize(const_cast<decayed_type *>(std::addressof(self))));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename... Args>
inline constexpr PetscErrorCode RegisterFinalizeable<D>::finalize_(Args &&...) noexcept
{
  return PETSC_SUCCESS;
}

template <typename D>
template <typename... Args>
inline constexpr PetscErrorCode RegisterFinalizeable<D>::register_finalize_(Args &&...) noexcept
{
  return PETSC_SUCCESS;
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
  PetscCall(do_finalize_(*this, std::forward<Args>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::finalize(Args &&...args) const noexcept
{
  PetscFunctionBegin;
  PetscCall(do_finalize_(*this, std::forward<Args>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscCall(do_register_finalize_(*this, std::forward<Args>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::register_finalize(Args &&...args) const noexcept
{
  PetscFunctionBegin;
  PetscCall(do_register_finalize_(*this, std::forward<Args>(args)...));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace Petsc
