#ifndef PETSCCUPMSOLVERINTERFACE_HPP
#define PETSCCUPMSOLVERINTERFACE_HPP

#if defined(__cplusplus)
  #include <petsc/private/cupmblasinterface.hpp>
  #include <petsc/private/petscadvancedmacros.h>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

  #define PetscCallCUPMSOLVER(...) \
    do { \
      const cupmSolverError_t cupmsolver_stat_p_ = __VA_ARGS__; \
      if (PetscUnlikely(cupmsolver_stat_p_ != CUPMSOLVER_STATUS_SUCCESS)) { \
        if (((cupmsolver_stat_p_ == CUPMSOLVER_STATUS_NOT_INITIALIZED) || (cupmsolver_stat_p_ == CUPMSOLVER_STATUS_ALLOC_FAILED) || (cupmsolver_stat_p_ == CUPMSOLVER_STATUS_INTERNAL_ERROR)) && PetscDeviceInitialized(PETSC_DEVICE_CUPM())) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "%s error %d (%s). " \
                  "This indicates the GPU may have run out resources", \
                  cupmSolverName(), static_cast<PetscErrorCode>(cupmsolver_stat_p_), cupmSolverGetErrorName(cupmsolver_stat_p_)); \
        } \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "%s error %d (%s)", cupmSolverName(), static_cast<PetscErrorCode>(cupmsolver_stat_p_), cupmSolverGetErrorName(cupmsolver_stat_p_)); \
      } \
    } while (0)

  #ifndef PetscConcat3
    #define PetscConcat3(a, b, c) PetscConcat(PetscConcat(a, b), c)
  #endif

  #if PetscDefined(USE_COMPLEX)
    #define PETSC_CUPMSOLVER_FP_TYPE_SPECIAL un
  #else
    #define PETSC_CUPMSOLVER_FP_TYPE_SPECIAL or
  #endif // USE_COMPLEX

  #define PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupm_name, their_prefix, fp_type, suffix) PETSC_CUPM_ALIAS_FUNCTION(cupm_name, PetscConcat3(their_prefix, fp_type, suffix))

template <DeviceType>
struct SolverInterfaceImpl;

  #if PetscDefined(HAVE_CUDA)
template <>
struct SolverInterfaceImpl<DeviceType::CUDA> : BlasInterface<DeviceType::CUDA> {
  // typedefs
  using cupmSolverHandle_t    = cusolverDnHandle_t;
  using cupmSolverError_t     = cusolverStatus_t;
  using cupmSolverFillMode_t  = cublasFillMode_t;
  using cupmSolverOperation_t = cublasOperation_t;

  // error codes
  static const auto CUPMSOLVER_STATUS_SUCCESS         = CUSOLVER_STATUS_SUCCESS;
  static const auto CUPMSOLVER_STATUS_NOT_INITIALIZED = CUSOLVER_STATUS_NOT_INITIALIZED;
  static const auto CUPMSOLVER_STATUS_ALLOC_FAILED    = CUSOLVER_STATUS_ALLOC_FAILED;
  static const auto CUPMSOLVER_STATUS_INTERNAL_ERROR  = CUSOLVER_STATUS_INTERNAL_ERROR;

  // enums
  // Why do these exist just to alias the CUBLAS versions? Because AMD -- in their boundless
  // wisdom -- decided to do so for hipSOLVER...
  // https://github.com/ROCmSoftwarePlatform/hipSOLVER/blob/develop/library/include/internal/hipsolver-types.h
  static const auto CUPMSOLVER_OP_T            = CUBLAS_OP_T;
  static const auto CUPMSOLVER_OP_N            = CUBLAS_OP_N;
  static const auto CUPMSOLVER_OP_C            = CUBLAS_OP_C;
  static const auto CUPMSOLVER_FILL_MODE_LOWER = CUBLAS_FILL_MODE_LOWER;
  static const auto CUPMSOLVER_FILL_MODE_UPPER = CUBLAS_FILL_MODE_UPPER;
  static const auto CUPMSOLVER_SIDE_LEFT       = CUBLAS_SIDE_LEFT;
  static const auto CUPMSOLVER_SIDE_RIGHT      = CUBLAS_SIDE_RIGHT;

  // utility functions
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverCreate, cusolverDnCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverDestroy, cusolverDnDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverSetStream, cusolverDnSetStream)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverGetStream, cusolverDnGetStream)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potrf)

  using cupmBlasInt_t = typename BlasInterface<DeviceType::CUDA>::cupmBlasInt_t;
  using cupmScalar_t  = typename Interface<DeviceType::CUDA>::cupmScalar_t;

  // to match hipSOLVER version (rocm 5.4.3, CUDA 12.0.1):
  //
  // hipsolverStatus_t hipsolverDpotrs_bufferSize(
  //   hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, double *A, int lda,
  //   double *B, int ldb, int *lwork
  // )
  //
  // hipsolverStatus_t hipsolverDpotrs(
  //   hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, int nrhs, double *A, int lda,
  //   double *B, int ldb, double *work, int lwork, int *devInfo
  // )
  PETSC_NODISCARD static constexpr cupmSolverError_t cupmSolverXpotrs_bufferSize(cupmSolverHandle_t /* handle */, cupmSolverFillMode_t /* uplo */, cupmBlasInt_t /* n */, cupmBlasInt_t /* nrhs */, cupmScalar_t * /* A */, cupmBlasInt_t /* lda */, cupmScalar_t * /* B */, cupmBlasInt_t /* ldb */, cupmBlasInt_t *lwork) noexcept
  {
    *lwork = 0;
    return CUPMSOLVER_STATUS_SUCCESS;
  }

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrs_p, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potrs)

  PETSC_NODISCARD static cupmSolverError_t cupmSolverXpotrs(cupmSolverHandle_t handle, cupmSolverFillMode_t uplo, cupmBlasInt_t n, cupmBlasInt_t nrhs, const cupmScalar_t *A, cupmBlasInt_t lda, cupmScalar_t *B, cupmBlasInt_t ldb, cupmScalar_t * /* work */, cupmBlasInt_t /* lwork */, cupmBlasInt_t *dev_info) noexcept
  {
    return cupmSolverXpotrs_p(handle, uplo, n, nrhs, A, lda, B, ldb, dev_info);
  }

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potri_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, potri)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, sytrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, sytrf)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, getrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf_p, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, getrf)
  // to match hipSOLVER version (rocm 5.4.3, CUDA 12.0.1):
  //
  // hipsolverStatus_t hipsolverDgetrf(
  //   hipsolverHandle_t handle, int m, int n, double *A, int lda, double *work, int lwork,
  //   int *devIpiv, int *devInfo
  // )
  PETSC_NODISCARD static cupmSolverError_t cupmSolverXgetrf(cupmSolverHandle_t handle, cupmBlasInt_t m, cupmBlasInt_t n, cupmScalar_t *A, cupmBlasInt_t lda, cupmScalar_t *work, cupmBlasInt_t /* lwork */, cupmBlasInt_t *dev_ipiv, cupmBlasInt_t *dev_info) noexcept
  {
    return cupmSolverXgetrf_p(handle, m, n, A, lda, work, dev_ipiv, dev_info);
  }

  // to match hipSOLVER version (rocm 5.4.3, CUDA 12.0.1):
  //
  // hipsolverStatus_t hipsolverDgetrs_bufferSize(
  //   hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, double *A,
  //   int lda, int *devIpiv, double *B, int ldb, int *lwork
  // )
  //
  // hipsolverStatus_t hipsolverDgetrs(
  //   hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, double *A,
  //   int lda, int *devIpiv, double *B, int ldb, double *work, int lwork, int *devInfo
  // )
  PETSC_NODISCARD static constexpr cupmSolverError_t cupmSolverXgetrs_bufferSize(cupmSolverHandle_t /* handle */, cupmSolverOperation_t /* op */, cupmBlasInt_t /* n */, cupmBlasInt_t /* nrhs */, cupmScalar_t * /* A */, cupmBlasInt_t /* lda */, cupmBlasInt_t * /* devIpiv */, cupmScalar_t * /* B */, cupmBlasInt_t /* ldb */, cupmBlasInt_t *lwork) noexcept
  {
    *lwork = 0;
    return CUPMSOLVER_STATUS_SUCCESS;
  }

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrs_p, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, getrs)

  PETSC_NODISCARD static cupmSolverError_t cupmSolverXgetrs(cupmSolverHandle_t handle, cupmSolverOperation_t op, cupmBlasInt_t n, cupmBlasInt_t nrhs, cupmScalar_t *A, cupmBlasInt_t lda, cupmBlasInt_t *dev_ipiv, cupmScalar_t *B, cupmBlasInt_t ldb, cupmScalar_t * /* work */, cupmBlasInt_t /* lwork */, cupmBlasInt_t *dev_info) noexcept
  {
    return cupmSolverXgetrs_p(handle, op, n, nrhs, A, lda, dev_ipiv, B, ldb, dev_info);
  }

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf_bufferSize, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, geqrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf, cusolverDn, PETSC_CUPMBLAS_FP_TYPE_U, geqrf)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr_bufferSize, cusolverDn, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr, cusolverDn, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr)

  PETSC_NODISCARD static const char *cupmSolverGetErrorName(cupmSolverError_t status) noexcept { return PetscCUSolverGetErrorName(status); }
};
  #endif

  #if PetscDefined(HAVE_HIP)
template <>
struct SolverInterfaceImpl<DeviceType::HIP> : BlasInterface<DeviceType::HIP> {
  // typedefs
  using cupmSolverHandle_t    = hipsolverHandle_t;
  using cupmSolverError_t     = hipsolverStatus_t;
  using cupmSolverFillMode_t  = hipsolverFillMode_t;
  using cupmSolverOperation_t = hipsolverOperation_t;

  // error codes
  static const auto CUPMSOLVER_STATUS_SUCCESS         = HIPSOLVER_STATUS_SUCCESS;
  static const auto CUPMSOLVER_STATUS_NOT_INITIALIZED = HIPSOLVER_STATUS_NOT_INITIALIZED;
  static const auto CUPMSOLVER_STATUS_ALLOC_FAILED    = HIPSOLVER_STATUS_ALLOC_FAILED;
  static const auto CUPMSOLVER_STATUS_INTERNAL_ERROR  = HIPSOLVER_STATUS_INTERNAL_ERROR;

  // enums
  static const auto CUPMSOLVER_OP_T            = HIPSOLVER_OP_T;
  static const auto CUPMSOLVER_OP_N            = HIPSOLVER_OP_N;
  static const auto CUPMSOLVER_OP_C            = HIPSOLVER_OP_C;
  static const auto CUPMSOLVER_FILL_MODE_LOWER = HIPSOLVER_FILL_MODE_LOWER;
  static const auto CUPMSOLVER_FILL_MODE_UPPER = HIPSOLVER_FILL_MODE_UPPER;
  static const auto CUPMSOLVER_SIDE_LEFT       = HIPSOLVER_SIDE_LEFT;
  static const auto CUPMSOLVER_SIDE_RIGHT      = HIPSOLVER_SIDE_RIGHT;

  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverCreate, hipsolverCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverDestroy, hipsolverDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverSetStream, hipsolverSetStream)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSolverGetStream, hipsolverGetStream)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potrf)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrs_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potrs_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotrs, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potrs)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potri_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXpotri, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, potri)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, sytrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXsytrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, sytrf)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, getrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, getrf)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrs_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, getrs_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgetrs, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, getrs)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf_bufferSize, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, geqrf_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXgeqrf, hipsolver, PETSC_CUPMBLAS_FP_TYPE_U, geqrf)

  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr_bufferSize, hipsolver, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr_bufferSize)
  PETSC_CUPMSOLVER_ALIAS_BLAS_FUNCTION(cupmSolverXormqr, hipsolver, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_U, PETSC_CUPMSOLVER_FP_TYPE_SPECIAL), mqr)

  PETSC_NODISCARD static const char *cupmSolverGetErrorName(cupmSolverError_t status) noexcept { return PetscHIPSolverGetErrorName(status); }
};
  #endif

  #define PETSC_CUPMSOLVER_IMPL_CLASS_HEADER(T) \
    PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(T); \
    /* introspection */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverGetErrorName; \
    /* types */ \
    using cupmSolverHandle_t    = typename ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverHandle_t; \
    using cupmSolverError_t     = typename ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverError_t; \
    using cupmSolverFillMode_t  = typename ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverFillMode_t; \
    using cupmSolverOperation_t = typename ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverOperation_t; \
    /* error codes */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_SUCCESS; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_NOT_INITIALIZED; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_ALLOC_FAILED; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_STATUS_INTERNAL_ERROR; \
    /* values */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_OP_T; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_OP_N; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_OP_C; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_FILL_MODE_LOWER; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_FILL_MODE_UPPER; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_SIDE_LEFT; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::CUPMSOLVER_SIDE_RIGHT; \
    /* utility functions */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverCreate; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverDestroy; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverGetStream; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverSetStream; \
    /* blas functions */ \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotrs_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotrs; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotri_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXpotri; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXsytrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXsytrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgetrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgetrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgetrs_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgetrs; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgeqrf_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXgeqrf; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXormqr_bufferSize; \
    using ::Petsc::device::cupm::impl::SolverInterfaceImpl<T>::cupmSolverXormqr

template <DeviceType T>
struct SolverInterface : SolverInterfaceImpl<T> {
  PETSC_NODISCARD static constexpr const char *cupmSolverName() noexcept { return T == DeviceType::CUDA ? "cusolverDn" : "hipsolver"; }
};

  #define PETSC_CUPMSOLVER_INHERIT_INTERFACE_TYPEDEFS_USING(T) \
    PETSC_CUPMSOLVER_IMPL_CLASS_HEADER(T); \
    using ::Petsc::device::cupm::impl::SolverInterface<T>::cupmSolverName

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCCUPMSOLVERINTERFACE_HPP
