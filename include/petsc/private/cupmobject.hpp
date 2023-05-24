#ifndef PETSC_PRIVATE_CUPMOBJECT_HPP
#define PETSC_PRIVATE_CUPMOBJECT_HPP

#ifdef __cplusplus
  #include <petsc/private/deviceimpl.h>
  #include <petsc/private/cupmsolverinterface.hpp>

  #include <cstring> // std::memset

namespace
{

inline PetscErrorCode PetscStrFreeAllocpy(const char target[], char **dest) noexcept
{
  PetscFunctionBegin;
  PetscValidPointer(dest, 2);
  if (*dest) {
    PetscValidCharPointer(*dest, 2);
    PetscCall(PetscFree(*dest));
  }
  PetscCall(PetscStrallocpy(target, dest));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

namespace
{

// ==========================================================================================
// UseCUPMHostAllocGuard
//
// A simple RAII helper for PetscMallocSet[CUDA|HIP]Host(). it exists because integrating the
// regular versions would be an enormous pain to square with the templated types...
// ==========================================================================================
template <DeviceType T>
class UseCUPMHostAllocGuard : Interface<T> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  UseCUPMHostAllocGuard(bool) noexcept;
  ~UseCUPMHostAllocGuard() noexcept;

  PETSC_NODISCARD bool value() const noexcept;

private:
    // would have loved to just do
    //
    // const auto oldmalloc = PetscTrMalloc;
    //
    // but in order to use auto the member needs to be static; in order to be static it must
    // also be constexpr -- which in turn requires an initializer (also implicitly required by
    // auto). But constexpr needs a constant expression initializer, so we can't initialize it
    // with global (mutable) variables...
  #define DECLTYPE_AUTO(left, right) decltype(right) left = right
  const DECLTYPE_AUTO(oldmalloc_, PetscTrMalloc);
  const DECLTYPE_AUTO(oldfree_, PetscTrFree);
  const DECLTYPE_AUTO(oldrealloc_, PetscTrRealloc);
  #undef DECLTYPE_AUTO
  bool v_;
};

// ==========================================================================================
// UseCUPMHostAllocGuard -- Public API
// ==========================================================================================

template <DeviceType T>
inline UseCUPMHostAllocGuard<T>::UseCUPMHostAllocGuard(bool useit) noexcept : v_(useit)
{
  PetscFunctionBegin;
  if (useit) {
    // all unused arguments are un-named, this saves having to add PETSC_UNUSED to them all
    PetscTrMalloc = [](std::size_t sz, PetscBool clear, int, const char *, const char *, void **ptr) {
      PetscFunctionBegin;
      PetscCallCUPM(cupmMallocHost(ptr, sz));
      if (clear) std::memset(*ptr, 0, sz);
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscTrFree = [](void *ptr, int, const char *, const char *) {
      PetscFunctionBegin;
      PetscCallCUPM(cupmFreeHost(ptr));
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    PetscTrRealloc = [](std::size_t, int, const char *, const char *, void **) {
      // REVIEW ME: can be implemented by malloc->copy->free?
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "%s has no realloc()", cupmName());
    };
  }
  PetscFunctionReturnVoid();
}

template <DeviceType T>
inline UseCUPMHostAllocGuard<T>::~UseCUPMHostAllocGuard() noexcept
{
  PetscFunctionBegin;
  if (value()) {
    PetscTrMalloc  = oldmalloc_;
    PetscTrFree    = oldfree_;
    PetscTrRealloc = oldrealloc_;
  }
  PetscFunctionReturnVoid();
}

template <DeviceType T>
inline bool UseCUPMHostAllocGuard<T>::value() const noexcept
{
  return v_;
}

} // anonymous namespace

template <DeviceType T, PetscMemType MemoryType, PetscMemoryAccessMode AccessMode>
class RestoreableArray : Interface<T> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  static constexpr auto memory_type = MemoryType;
  static constexpr auto access_type = AccessMode;

  using value_type        = PetscScalar;
  using pointer_type      = value_type *;
  using cupm_pointer_type = cupmScalar_t *;

  PETSC_NODISCARD pointer_type      data() const noexcept;
  PETSC_NODISCARD cupm_pointer_type cupmdata() const noexcept;

  operator pointer_type() const noexcept;
  // in case pointer_type == cupmscalar_pointer_type we don't want this overload to exist, so
  // we make a dummy template parameter to allow SFINAE to nix it for us
  template <typename U = pointer_type, typename = util::enable_if_t<!std::is_same<U, cupm_pointer_type>::value>>
  operator cupm_pointer_type() const noexcept;

protected:
  constexpr explicit RestoreableArray(PetscDeviceContext) noexcept;

  value_type        *ptr_  = nullptr;
  PetscDeviceContext dctx_ = nullptr;
};

// ==========================================================================================
// RestoreableArray - Static Variables
// ==========================================================================================

template <DeviceType T, PetscMemType MT, PetscMemoryAccessMode MA>
const PetscMemType RestoreableArray<T, MT, MA>::memory_type;

template <DeviceType T, PetscMemType MT, PetscMemoryAccessMode MA>
const PetscMemoryAccessMode RestoreableArray<T, MT, MA>::access_type;

// ==========================================================================================
// RestoreableArray - Public API
// ==========================================================================================

template <DeviceType T, PetscMemType MT, PetscMemoryAccessMode MA>
constexpr inline RestoreableArray<T, MT, MA>::RestoreableArray(PetscDeviceContext dctx) noexcept : dctx_{dctx}
{
}

template <DeviceType T, PetscMemType MT, PetscMemoryAccessMode MA>
inline typename RestoreableArray<T, MT, MA>::pointer_type RestoreableArray<T, MT, MA>::data() const noexcept
{
  return ptr_;
}

template <DeviceType T, PetscMemType MT, PetscMemoryAccessMode MA>
inline typename RestoreableArray<T, MT, MA>::cupm_pointer_type RestoreableArray<T, MT, MA>::cupmdata() const noexcept
{
  return cupmScalarPtrCast(data());
}

template <DeviceType T, PetscMemType MT, PetscMemoryAccessMode MA>
inline RestoreableArray<T, MT, MA>::operator pointer_type() const noexcept
{
  return data();
}

// in case pointer_type == cupmscalar_pointer_type we don't want this overload to exist, so
// we make a dummy template parameter to allow SFINAE to nix it for us
template <DeviceType T, PetscMemType MT, PetscMemoryAccessMode MA>
template <typename U, typename>
inline RestoreableArray<T, MT, MA>::operator cupm_pointer_type() const noexcept
{
  return cupmdata();
}

template <DeviceType T>
class CUPMObject : SolverInterface<T> {
protected:
  PETSC_CUPMSOLVER_INHERIT_INTERFACE_TYPEDEFS_USING(T);

private:
  // The final stop in the GetHandles_/GetFromHandles_ chain. This retrieves the various
  // compute handles and ensure the given PetscDeviceContext is of the right type
  static PetscErrorCode GetFromHandleDispatch_(PetscDeviceContext, cupmBlasHandle_t *, cupmSolverHandle_t *, cupmStream_t *) noexcept;
  static PetscErrorCode GetHandleDispatch_(PetscDeviceContext *, cupmBlasHandle_t *, cupmSolverHandle_t *, cupmStream_t *) noexcept;

protected:
  PETSC_NODISCARD static constexpr PetscRandomType PETSCDEVICERAND() noexcept;

  // Helper routines to retrieve various combinations of handles. The first set (GetHandles_)
  // gets a PetscDeviceContext along with it, while the second set (GetHandlesFrom_) assumes
  // you've gotten the PetscDeviceContext already, and retrieves the handles from it. All of
  // them check that the PetscDeviceContext is of the appropriate type
  static PetscErrorCode GetHandles_(PetscDeviceContext *, cupmBlasHandle_t * = nullptr, cupmSolverHandle_t * = nullptr, cupmStream_t * = nullptr) noexcept;

  // triple
  static PetscErrorCode GetHandles_(PetscDeviceContext *, cupmBlasHandle_t *, cupmStream_t *) noexcept;
  static PetscErrorCode GetHandles_(PetscDeviceContext *, cupmSolverHandle_t *, cupmStream_t *) noexcept;

  // double
  static PetscErrorCode GetHandles_(PetscDeviceContext *, cupmSolverHandle_t *) noexcept;
  static PetscErrorCode GetHandles_(PetscDeviceContext *, cupmStream_t *) noexcept;

  // single
  static PetscErrorCode GetHandles_(cupmBlasHandle_t *) noexcept;
  static PetscErrorCode GetHandles_(cupmSolverHandle_t *) noexcept;
  static PetscErrorCode GetHandles_(cupmStream_t *) noexcept;

  static PetscErrorCode GetHandlesFrom_(PetscDeviceContext, cupmBlasHandle_t *, cupmSolverHandle_t * = nullptr, cupmStream_t * = nullptr) noexcept;
  static PetscErrorCode GetHandlesFrom_(PetscDeviceContext, cupmSolverHandle_t *, cupmStream_t * = nullptr) noexcept;
  static PetscErrorCode GetHandlesFrom_(PetscDeviceContext, cupmStream_t *) noexcept;

  // disallow implicit conversion
  template <typename U>
  PETSC_NODISCARD static UseCUPMHostAllocGuard<T> UseCUPMHostAlloc(U) noexcept = delete;
  // utility for using cupmHostAlloc()
  PETSC_NODISCARD static UseCUPMHostAllocGuard<T> UseCUPMHostAlloc(bool) noexcept;
  PETSC_NODISCARD static UseCUPMHostAllocGuard<T> UseCUPMHostAlloc(PetscBool) noexcept;

  // A debug check to ensure that a given pointer-memtype pairing taken from user-land is
  // actually correct. Errors on mismatch
  static PetscErrorCode CheckPointerMatchesMemType_(const void *, PetscMemType) noexcept;
};

template <DeviceType T>
inline constexpr PetscRandomType CUPMObject<T>::PETSCDEVICERAND() noexcept
{
  // REVIEW ME: HIP default rng?
  return T == DeviceType::CUDA ? PETSCCURAND : PETSCRANDER48;
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetFromHandleDispatch_(PetscDeviceContext dctx, cupmBlasHandle_t *blas_handle, cupmSolverHandle_t *solver_handle, cupmStream_t *stream) noexcept
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  if (blas_handle) PetscValidPointer(blas_handle, 2);
  if (solver_handle) PetscValidPointer(solver_handle, 3);
  if (stream) PetscValidPointer(stream, 4);
  if (PetscDefined(USE_DEBUG)) {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCheckCompatibleDeviceTypes(PETSC_DEVICE_CUPM(), -1, dtype, 1);
  }
  if (blas_handle) PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, blas_handle));
  if (solver_handle) PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(dctx, solver_handle));
  if (stream) PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, stream));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandleDispatch_(PetscDeviceContext *dctx, cupmBlasHandle_t *blas_handle, cupmSolverHandle_t *solver_handle, cupmStream_t *stream) noexcept
{
  PetscDeviceContext dctx_loc = nullptr;

  PetscFunctionBegin;
  // silence uninitialized variable warnings
  if (dctx) *dctx = nullptr;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx_loc));
  PetscCall(GetFromHandleDispatch_(dctx_loc, blas_handle, solver_handle, stream));
  if (dctx) *dctx = dctx_loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandles_(PetscDeviceContext *dctx, cupmBlasHandle_t *blas_handle, cupmSolverHandle_t *solver_handle, cupmStream_t *stream) noexcept
{
  return GetHandleDispatch_(dctx, blas_handle, solver_handle, stream);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandles_(PetscDeviceContext *dctx, cupmBlasHandle_t *blas_handle, cupmStream_t *stream) noexcept
{
  return GetHandleDispatch_(dctx, blas_handle, nullptr, stream);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandles_(PetscDeviceContext *dctx, cupmSolverHandle_t *solver_handle, cupmStream_t *stream) noexcept
{
  return GetHandleDispatch_(dctx, nullptr, solver_handle, stream);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandles_(PetscDeviceContext *dctx, cupmStream_t *stream) noexcept
{
  return GetHandleDispatch_(dctx, nullptr, nullptr, stream);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandles_(cupmBlasHandle_t *handle) noexcept
{
  return GetHandleDispatch_(nullptr, handle, nullptr, nullptr);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandles_(cupmSolverHandle_t *handle) noexcept
{
  return GetHandleDispatch_(nullptr, nullptr, handle, nullptr);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandles_(cupmStream_t *stream) noexcept
{
  return GetHandleDispatch_(nullptr, nullptr, nullptr, stream);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandlesFrom_(PetscDeviceContext dctx, cupmBlasHandle_t *blas_handle, cupmSolverHandle_t *solver_handle, cupmStream_t *stream) noexcept
{
  return GetFromHandleDispatch_(dctx, blas_handle, solver_handle, stream);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandlesFrom_(PetscDeviceContext dctx, cupmSolverHandle_t *solver_handle, cupmStream_t *stream) noexcept
{
  return GetFromHandleDispatch_(dctx, nullptr, solver_handle, stream);
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::GetHandlesFrom_(PetscDeviceContext dctx, cupmStream_t *stream) noexcept
{
  return GetFromHandleDispatch_(dctx, nullptr, nullptr, stream);
}

template <DeviceType T>
inline UseCUPMHostAllocGuard<T> CUPMObject<T>::UseCUPMHostAlloc(bool b) noexcept
{
  return {b};
}

template <DeviceType T>
inline UseCUPMHostAllocGuard<T> CUPMObject<T>::UseCUPMHostAlloc(PetscBool b) noexcept
{
  return UseCUPMHostAlloc(static_cast<bool>(b));
}

template <DeviceType T>
inline PetscErrorCode CUPMObject<T>::CheckPointerMatchesMemType_(const void *ptr, PetscMemType mtype) noexcept
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG) && ptr) {
    PetscMemType ptr_mtype;

    PetscCall(PetscCUPMGetMemType(ptr, &ptr_mtype));
    if (mtype == PETSC_MEMTYPE_HOST) {
      PetscCheck(PetscMemTypeHost(ptr_mtype), PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p declared as %s does not match actual memtype %s", ptr, PetscMemTypeToString(mtype), PetscMemTypeToString(ptr_mtype));
    } else if (mtype == PETSC_MEMTYPE_DEVICE) {
      // generic "device" memory should only care if the actual memtype is also generically
      // "device"
      PetscCheck(PetscMemTypeDevice(ptr_mtype), PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p declared as %s does not match actual memtype %s", ptr, PetscMemTypeToString(mtype), PetscMemTypeToString(ptr_mtype));
    } else {
      PetscCheck(mtype == ptr_mtype, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p declared as %s does not match actual memtype %s", ptr, PetscMemTypeToString(mtype), PetscMemTypeToString(ptr_mtype));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define PETSC_CUPMOBJECT_HEADER(T) \
    PETSC_CUPMSOLVER_INHERIT_INTERFACE_TYPEDEFS_USING(T); \
    using ::Petsc::device::cupm::impl::CUPMObject<T>::UseCUPMHostAlloc; \
    using ::Petsc::device::cupm::impl::CUPMObject<T>::GetHandles_; \
    using ::Petsc::device::cupm::impl::CUPMObject<T>::GetHandlesFrom_; \
    using ::Petsc::device::cupm::impl::CUPMObject<T>::PETSCDEVICERAND; \
    using ::Petsc::device::cupm::impl::CUPMObject<T>::CheckPointerMatchesMemType_

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_PRIVATE_CUPMOBJECT_HPP
