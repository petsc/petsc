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

template <typename Derived>
class RegisterFinalizeable : public util::crtp<Derived, RegisterFinalizeable> {
public:
  using derived_type = Derived;
  using crtp_type    = util::crtp<Derived, RegisterFinalizeable>;

  PETSC_NODISCARD bool registered() const noexcept;
  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode finalize(Args &&...) noexcept;
  template <typename... Args>
  PETSC_NODISCARD PetscErrorCode register_finalize(MPI_Comm comm = PETSC_COMM_SELF, Args &&...) noexcept;

private:
  constexpr RegisterFinalizeable() = default;
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

template <typename D>
inline bool RegisterFinalizeable<D>::registered() const noexcept
{
  return registered_;
}

template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::finalize(Args &&...args) noexcept
{
  PetscFunctionBegin;
  // order of registered_ matters here, if the finalizer wants to re-register it should be able to
  registered_ = false;
  PetscCall(this->underlying().finalize_(std::forward<Args>(args)...));
  PetscFunctionReturn(0);
}

template <typename D>
template <typename... Args>
inline PetscErrorCode RegisterFinalizeable<D>::register_finalize(MPI_Comm comm, Args &&...args) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(this->underlying().registered())) PetscFunctionReturn(0);
  registered_ = true;
  PetscCall(this->underlying().register_finalize_(std::forward<Args>(args)...));
  PetscCall(PetscCxxObjectRegisterFinalize(this, comm));
  PetscFunctionReturn(0);
}

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CPP_REGISTER_FINALIZE_HPP
