#pragma once

#include <petsc/private/cupmblasinterface.hpp>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

#define PetscCallCUPMSPARSE_(__abort_fn__, __comm__, ...) \
  do { \
    PetscStackUpdateLine; \
    const cupmSparseError_t cupmsparse_stat_p_ = __VA_ARGS__; \
    if (PetscUnlikely(cupmsparse_stat_p_ != CUPMSPARSE_STATUS_SUCCESS)) { \
      if (cupmsparse_stat_p_ == CUPMSPARSE_STATUS_NOT_INITIALIZED || cupmsparse_stat_p_ == CUPMSPARSE_STATUS_ALLOC_FAILED) \
        __abort_fn__(__comm__, PETSC_ERR_GPU_RESOURCE, "%s error %d (%s)", cupmSparseName(), static_cast<PetscErrorCode>(cupmsparse_stat_p_), cupmSparseGetErrorName(cupmsparse_stat_p_)); \
      __abort_fn__(__comm__, PETSC_ERR_GPU, "%s error %d (%s)", cupmSparseName(), static_cast<PetscErrorCode>(cupmsparse_stat_p_), cupmSparseGetErrorName(cupmsparse_stat_p_)); \
    } \
  } while (0)

#define PetscCallCUPMSPARSE(...)             PetscCallCUPMSPARSE_(SETERRQ, PETSC_COMM_SELF, __VA_ARGS__)
#define PetscCallCUPMSPARSEAbort(comm_, ...) PetscCallCUPMSPARSE_(SETERRABORT, comm_, __VA_ARGS__)

template <DeviceType>
struct SparseInterface;

#if PetscDefined(HAVE_CUDA)
template <>
struct PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL SparseInterface<DeviceType::CUDA> : BlasInterface<DeviceType::CUDA> {
  using cupmSparseHandle_t                                = cusparseHandle_t;
  using cupmSparseError_t                                 = cusparseStatus_t;
  using cupmSparsePointerMode_t                           = cusparsePointerMode_t;
  static constexpr auto CUPMSPARSE_STATUS_SUCCESS         = CUSPARSE_STATUS_SUCCESS;
  static constexpr auto CUPMSPARSE_STATUS_NOT_INITIALIZED = CUSPARSE_STATUS_NOT_INITIALIZED;
  static constexpr auto CUPMSPARSE_STATUS_ALLOC_FAILED    = CUSPARSE_STATUS_ALLOC_FAILED;
  static constexpr auto CUPMSPARSE_POINTER_MODE_HOST      = CUSPARSE_POINTER_MODE_HOST;
  static constexpr auto CUPMSPARSE_POINTER_MODE_DEVICE    = CUSPARSE_POINTER_MODE_DEVICE;
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseCreate, cusparseCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseDestroy, cusparseDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseSetStream, cusparseSetStream)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseGetPointerMode, cusparseGetPointerMode)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseSetPointerMode, cusparseSetPointerMode)
  PETSC_NODISCARD static const char *cupmSparseName() noexcept { return "cuSPARSE"; }
  PETSC_NODISCARD static const char *cupmSparseGetErrorName(cupmSparseError_t status) noexcept { return cusparseGetErrorName(status); }
};
#endif

#if PetscDefined(HAVE_HIP)
template <>
struct PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL SparseInterface<DeviceType::HIP> : BlasInterface<DeviceType::HIP> {
  using cupmSparseHandle_t                                = hipsparseHandle_t;
  using cupmSparseError_t                                 = hipsparseStatus_t;
  using cupmSparsePointerMode_t                           = hipsparsePointerMode_t;
  static constexpr auto CUPMSPARSE_STATUS_SUCCESS         = HIPSPARSE_STATUS_SUCCESS;
  static constexpr auto CUPMSPARSE_STATUS_NOT_INITIALIZED = HIPSPARSE_STATUS_NOT_INITIALIZED;
  static constexpr auto CUPMSPARSE_STATUS_ALLOC_FAILED    = HIPSPARSE_STATUS_ALLOC_FAILED;
  static constexpr auto CUPMSPARSE_POINTER_MODE_HOST      = HIPSPARSE_POINTER_MODE_HOST;
  static constexpr auto CUPMSPARSE_POINTER_MODE_DEVICE    = HIPSPARSE_POINTER_MODE_DEVICE;
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseCreate, hipsparseCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseDestroy, hipsparseDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseSetStream, hipsparseSetStream)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseGetPointerMode, hipsparseGetPointerMode)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSparseSetPointerMode, hipsparseSetPointerMode)
  PETSC_NODISCARD static const char *cupmSparseName() noexcept { return "hipSPARSE"; }
  PETSC_NODISCARD static const char *cupmSparseGetErrorName(cupmSparseError_t status) noexcept { return PetscHIPSPARSEGetErrorName(status); }
};
#endif

#define PETSC_CUPMSPARSE_INHERIT_INTERFACE_TYPEDEFS_USING(T) \
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(T); \
  using cupmSparseHandle_t      = typename ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseHandle_t; \
  using cupmSparseError_t       = typename ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseError_t; \
  using cupmSparsePointerMode_t = typename ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparsePointerMode_t; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::CUPMSPARSE_STATUS_SUCCESS; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::CUPMSPARSE_STATUS_NOT_INITIALIZED; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::CUPMSPARSE_STATUS_ALLOC_FAILED; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::CUPMSPARSE_POINTER_MODE_HOST; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::CUPMSPARSE_POINTER_MODE_DEVICE; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseCreate; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseDestroy; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseSetStream; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseGetPointerMode; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseSetPointerMode; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseName; \
  using ::Petsc::device::cupm::impl::SparseInterface<T>::cupmSparseGetErrorName

template <DeviceType T>
class PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL CUPMSparsePointerModeGuard : SparseInterface<T> {
  PETSC_CUPMSPARSE_INHERIT_INTERFACE_TYPEDEFS_USING(T);

public:
  CUPMSparsePointerModeGuard(const cupmSparseHandle_t &handle, cupmSparsePointerMode_t mode) noexcept : handle_{handle}
  {
    PetscFunctionBegin;
    PetscCallCUPMSPARSEAbort(PETSC_COMM_SELF, cupmSparseGetPointerMode(handle, &this->saved_));
    PetscCallCUPMSPARSEAbort(PETSC_COMM_SELF, cupmSparseSetPointerMode(handle, mode));
    PetscFunctionReturnVoid();
  }

  CUPMSparsePointerModeGuard(const CUPMSparsePointerModeGuard &)            = delete;
  CUPMSparsePointerModeGuard &operator=(const CUPMSparsePointerModeGuard &) = delete;

  ~CUPMSparsePointerModeGuard() noexcept
  {
    PetscFunctionBegin;
    PetscCallCUPMSPARSEAbort(PETSC_COMM_SELF, cupmSparseSetPointerMode(this->handle_, this->saved_));
    PetscFunctionReturnVoid();
  }

private:
  cupmSparseHandle_t      handle_;
  cupmSparsePointerMode_t saved_;
};

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
