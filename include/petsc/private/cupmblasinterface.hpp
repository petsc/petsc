#ifndef PETSCCUPMBLASINTERFACE_HPP
#define PETSCCUPMBLASINTERFACE_HPP

#if defined(__cplusplus)
  #include <petsc/private/cupminterface.hpp>
  #include <petsc/private/petscadvancedmacros.h>

  #include <limits> // std::numeric_limits

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

  #define PetscCallCUPMBLAS(...) \
    do { \
      const cupmBlasError_t cberr_p_ = __VA_ARGS__; \
      if (PetscUnlikely(cberr_p_ != CUPMBLAS_STATUS_SUCCESS)) { \
        if (((cberr_p_ == CUPMBLAS_STATUS_NOT_INITIALIZED) || (cberr_p_ == CUPMBLAS_STATUS_ALLOC_FAILED)) && PetscDeviceInitialized(PETSC_DEVICE_CUPM())) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "%s error %d (%s). Reports not initialized or alloc failed; " \
                  "this indicates the GPU may have run out resources", \
                  cupmBlasName(), static_cast<PetscErrorCode>(cberr_p_), cupmBlasGetErrorName(cberr_p_)); \
        } \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "%s error %d (%s)", cupmBlasName(), static_cast<PetscErrorCode>(cberr_p_), cupmBlasGetErrorName(cberr_p_)); \
      } \
    } while (0)

  #define PetscCallCUPMBLASAbort(comm, ...) \
    do { \
      const cupmBlasError_t cberr_abort_p_ = __VA_ARGS__; \
      if (PetscUnlikely(cberr_abort_p_ != CUPMBLAS_STATUS_SUCCESS)) { \
        if (((cberr_abort_p_ == CUPMBLAS_STATUS_NOT_INITIALIZED) || (cberr_abort_p_ == CUPMBLAS_STATUS_ALLOC_FAILED)) && PetscDeviceInitialized(PETSC_DEVICE_CUPM())) { \
          SETERRABORT(comm, PETSC_ERR_GPU_RESOURCE, \
                      "%s error %d (%s). Reports not initialized or alloc failed; " \
                      "this indicates the GPU may have run out resources", \
                      cupmBlasName(), static_cast<PetscErrorCode>(cberr_abort_p_), cupmBlasGetErrorName(cberr_abort_p_)); \
        } \
        SETERRABORT(comm, PETSC_ERR_GPU, "%s error %d (%s)", cupmBlasName(), static_cast<PetscErrorCode>(cberr_abort_p_), cupmBlasGetErrorName(cberr_abort_p_)); \
      } \
    } while (0)

  // given cupmBlas<T>axpy() then
  // T = PETSC_CUPBLAS_FP_TYPE
  // given cupmBlas<T><u>nrm2() then
  // T = PETSC_CUPMBLAS_FP_INPUT_TYPE
  // u = PETSC_CUPMBLAS_FP_RETURN_TYPE
  #if PetscDefined(USE_COMPLEX)
    #if PetscDefined(USE_REAL_SINGLE)
      #define PETSC_CUPMBLAS_FP_TYPE_U       C
      #define PETSC_CUPMBLAS_FP_TYPE_L       c
      #define PETSC_CUPMBLAS_FP_INPUT_TYPE_U S
      #define PETSC_CUPMBLAS_FP_INPUT_TYPE_L s
    #elif PetscDefined(USE_REAL_DOUBLE)
      #define PETSC_CUPMBLAS_FP_TYPE_U       Z
      #define PETSC_CUPMBLAS_FP_TYPE_L       z
      #define PETSC_CUPMBLAS_FP_INPUT_TYPE_U D
      #define PETSC_CUPMBLAS_FP_INPUT_TYPE_L d
    #endif
    #define PETSC_CUPMBLAS_FP_RETURN_TYPE_U PETSC_CUPMBLAS_FP_TYPE_U
    #define PETSC_CUPMBLAS_FP_RETURN_TYPE_L PETSC_CUPMBLAS_FP_TYPE_L
  #else
    #if PetscDefined(USE_REAL_SINGLE)
      #define PETSC_CUPMBLAS_FP_TYPE_U S
      #define PETSC_CUPMBLAS_FP_TYPE_L s
    #elif PetscDefined(USE_REAL_DOUBLE)
      #define PETSC_CUPMBLAS_FP_TYPE_U D
      #define PETSC_CUPMBLAS_FP_TYPE_L d
    #endif
    #define PETSC_CUPMBLAS_FP_INPUT_TYPE_U PETSC_CUPMBLAS_FP_TYPE_U
    #define PETSC_CUPMBLAS_FP_INPUT_TYPE_L PETSC_CUPMBLAS_FP_TYPE_L
    #define PETSC_CUPMBLAS_FP_RETURN_TYPE_U
    #define PETSC_CUPMBLAS_FP_RETURN_TYPE_L
  #endif // USE_COMPLEX

  #if !defined(PETSC_CUPMBLAS_FP_TYPE_U) && !PetscDefined(USE_REAL___FLOAT128)
    #error "Unsupported floating-point type for CUDA/HIP BLAS"
  #endif

  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_MODIFIED() - Helper macro to build a "modified"
  // blas function whose return type does not match the input type
  //
  // input param:
  // func - base suffix of the blas function, e.g. nrm2
  //
  // notes:
  // requires PETSC_CUPMBLAS_FP_INPUT_TYPE to be defined as the blas floating point input type
  // letter ("S" for real/complex single, "D" for real/complex double).
  //
  // requires PETSC_CUPMBLAS_FP_RETURN_TYPE to be defined as the blas floating point output type
  // letter ("c" for complex single, "z" for complex double and <absolutely nothing> for real
  // single/double).
  //
  // In their infinite wisdom nvidia/amd have made the upper-case vs lower-case scheme
  // infuriatingly inconsistent...
  //
  // example usage:
  // #define PETSC_CUPMBLAS_FP_INPUT_TYPE  S
  // #define PETSC_CUPMBLAS_FP_RETURN_TYPE
  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_MODIFIED(nrm2) -> Snrm2
  //
  // #define PETSC_CUPMBLAS_FP_INPUT_TYPE  D
  // #define PETSC_CUPMBLAS_FP_RETURN_TYPE z
  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_MODIFIED(nrm2) -> Dznrm2
  #define PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_MODIFIED(func) PetscConcat(PetscConcat(PETSC_CUPMBLAS_FP_INPUT_TYPE, PETSC_CUPMBLAS_FP_RETURN_TYPE), func)

  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_IFPTYPE() - Helper macro to build Iamax and Iamin
  // because they are both extra special
  //
  // input param:
  // func - base suffix of the blas function, either amax or amin
  //
  // notes:
  // The macro name literally stands for "I" ## "floating point type" because shockingly enough,
  // that's what it does.
  //
  // requires PETSC_CUPMBLAS_FP_TYPE_L to be defined as the lower-case blas floating point input type
  // letter ("s" for complex single, "z" for complex double, "s" for real single, and "d" for
  // real double).
  //
  // example usage:
  // #define PETSC_CUPMBLAS_FP_TYPE_L s
  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_IFPTYPE(amax) -> Isamax
  //
  // #define PETSC_CUPMBLAS_FP_TYPE_L z
  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_IFPTYPE(amin) -> Izamin
  #define PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_IFPTYPE(func) PetscConcat(I, PetscConcat(PETSC_CUPMBLAS_FP_TYPE_L, func))

  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_STANDARD() - Helper macro to build a "standard"
  // blas function name
  //
  // input param:
  // func - base suffix of the blas function, e.g. axpy, scal
  //
  // notes:
  // requires PETSC_CUPMBLAS_FP_TYPE to be defined as the blas floating-point letter ("C" for
  // complex single, "Z" for complex double, "S" for real single, "D" for real double).
  //
  // example usage:
  // #define PETSC_CUPMBLAS_FP_TYPE S
  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_STANDARD(axpy) -> Saxpy
  //
  // #define PETSC_CUPMBLAS_FP_TYPE Z
  // PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_STANDARD(axpy) -> Zaxpy
  #define PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_STANDARD(func) PetscConcat(PETSC_CUPMBLAS_FP_TYPE, func)

  // PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT() - In case CUDA/HIP don't agree with our suffix
  // one can provide both here
  //
  // input params:
  // MACRO_SUFFIX - suffix to one of the above blas function builder macros, e.g. STANDARD or
  // IFPTYPE
  // our_suffix   - the suffix of the alias function
  // their_suffix - the suffix of the function being aliased
  //
  // notes:
  // requires PETSC_CUPMBLAS_PREFIX to be defined as the specific CUDA/HIP blas function
  // prefix. requires any other specific definitions required by the specific builder macro to
  // also be defined. See PETSC_CUPM_ALIAS_FUNCTION_EXACT() for the exact expansion of the
  // function alias.
  //
  // example usage:
  // #define PETSC_CUPMBLAS_PREFIX  cublas
  // #define PETSC_CUPMBLAS_FP_TYPE C
  // PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD,dot,dotc) ->
  // template <typename... T>
  // static constexpr auto cupmBlasXdot(T&&... args) *noexcept and returntype detection*
  // {
  //   return cublasCdotc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(MACRO_SUFFIX, our_suffix, their_suffix) \
    PETSC_CUPM_ALIAS_FUNCTION(PetscConcat(cupmBlasX, our_suffix), PetscConcat(PETSC_CUPMBLAS_PREFIX, PetscConcat(PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_, MACRO_SUFFIX)(their_suffix)))

  // PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION() - Alias a CUDA/HIP blas function
  //
  // input params:
  // MACRO_SUFFIX - suffix to one of the above blas function builder macros, e.g. STANDARD or
  // IFPTYPE
  // suffix       - the common suffix between CUDA and HIP of the alias function
  //
  // notes:
  // see PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(), this macro just calls that one with "suffix" as
  // "our_prefix" and "their_prefix"
  #define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MACRO_SUFFIX, suffix) PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(MACRO_SUFFIX, suffix, suffix)

  // PETSC_CUPMBLAS_ALIAS_FUNCTION() - Alias a CUDA/HIP library function
  //
  // input params:
  // suffix - the common suffix between CUDA and HIP of the alias function
  //
  // notes:
  // requires PETSC_CUPMBLAS_PREFIX to be defined as the specific CUDA/HIP blas library
  // prefix. see PETSC_CUPMM_ALIAS_FUNCTION_EXACT() for the precise expansion of this macro.
  //
  // example usage:
  // #define PETSC_CUPMBLAS_PREFIX hipblas
  // PETSC_CUPMBLAS_ALIAS_FUNCTION(Create) ->
  // template <typename... T>
  // static constexpr auto cupmBlasCreate(T&&... args) *noexcept and returntype detection*
  // {
  //   return hipblasCreate(std::forward<T>(args)...);
  // }
  #define PETSC_CUPMBLAS_ALIAS_FUNCTION(suffix) PETSC_CUPM_ALIAS_FUNCTION(PetscConcat(cupmBlas, suffix), PetscConcat(PETSC_CUPMBLAS_PREFIX, suffix))

template <DeviceType>
struct BlasInterfaceImpl;

// Exists because HIP (for whatever godforsaken reason) has elected to define both their
// hipBlasHandle_t and hipSolverHandle_t as void *. So we cannot disambiguate them for overload
// resolution and hence need to wrap their types int this mess.
template <typename T, std::size_t I>
class cupmBlasHandleWrapper {
public:
  constexpr cupmBlasHandleWrapper() noexcept = default;
  constexpr cupmBlasHandleWrapper(T h) noexcept : handle_(std::move(h)) { static_assert(std::is_standard_layout<cupmBlasHandleWrapper<T, I>>::value, ""); }

  cupmBlasHandleWrapper &operator=(std::nullptr_t) noexcept
  {
    handle_ = nullptr;
    return *this;
  }

  operator T() const { return handle_; }

  const T *ptr_to() const { return &handle_; }
  T       *ptr_to() { return &handle_; }

private:
  T handle_{};
};

  #if PetscDefined(HAVE_CUDA)
    #define PETSC_CUPMBLAS_PREFIX         cublas
    #define PETSC_CUPMBLAS_PREFIX_U       CUBLAS
    #define PETSC_CUPMBLAS_FP_TYPE        PETSC_CUPMBLAS_FP_TYPE_U
    #define PETSC_CUPMBLAS_FP_INPUT_TYPE  PETSC_CUPMBLAS_FP_INPUT_TYPE_U
    #define PETSC_CUPMBLAS_FP_RETURN_TYPE PETSC_CUPMBLAS_FP_RETURN_TYPE_L
template <>
struct BlasInterfaceImpl<DeviceType::CUDA> : Interface<DeviceType::CUDA> {
  // typedefs
  using cupmBlasHandle_t      = cupmBlasHandleWrapper<cublasHandle_t, 0>;
  using cupmBlasError_t       = cublasStatus_t;
  using cupmBlasInt_t         = int;
  using cupmBlasPointerMode_t = cublasPointerMode_t;

  // values
  static const auto CUPMBLAS_STATUS_SUCCESS         = CUBLAS_STATUS_SUCCESS;
  static const auto CUPMBLAS_STATUS_NOT_INITIALIZED = CUBLAS_STATUS_NOT_INITIALIZED;
  static const auto CUPMBLAS_STATUS_ALLOC_FAILED    = CUBLAS_STATUS_ALLOC_FAILED;
  static const auto CUPMBLAS_POINTER_MODE_HOST      = CUBLAS_POINTER_MODE_HOST;
  static const auto CUPMBLAS_POINTER_MODE_DEVICE    = CUBLAS_POINTER_MODE_DEVICE;
  static const auto CUPMBLAS_OP_T                   = CUBLAS_OP_T;
  static const auto CUPMBLAS_OP_N                   = CUBLAS_OP_N;
  static const auto CUPMBLAS_OP_C                   = CUBLAS_OP_C;
  static const auto CUPMBLAS_FILL_MODE_LOWER        = CUBLAS_FILL_MODE_LOWER;
  static const auto CUPMBLAS_FILL_MODE_UPPER        = CUBLAS_FILL_MODE_UPPER;
  static const auto CUPMBLAS_SIDE_LEFT              = CUBLAS_SIDE_LEFT;
  static const auto CUPMBLAS_DIAG_NON_UNIT          = CUBLAS_DIAG_NON_UNIT;

  // utility functions
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Create)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Destroy)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(GetStream)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(SetStream)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(GetPointerMode)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(SetPointerMode)

  // level 1 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, axpy)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, scal)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD, dot, PetscIfPetscDefined(USE_COMPLEX, dotc, dot))
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD, dotu, PetscIfPetscDefined(USE_COMPLEX, dotu, dot))
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, swap)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED, nrm2)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(IFPTYPE, amax)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED, asum)

  // level 2 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, gemv)

  // level 3 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, gemm)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, trsm)

  // BLAS extensions
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, geam)

  PETSC_NODISCARD static const char *cupmBlasGetErrorName(cupmBlasError_t status) noexcept { return PetscCUBLASGetErrorName(status); }
};
    #undef PETSC_CUPMBLAS_PREFIX
    #undef PETSC_CUPMBLAS_PREFIX_U
    #undef PETSC_CUPMBLAS_FP_TYPE
    #undef PETSC_CUPMBLAS_FP_INPUT_TYPE
    #undef PETSC_CUPMBLAS_FP_RETURN_TYPE
  #endif // PetscDefined(HAVE_CUDA)

  #if PetscDefined(HAVE_HIP)
    #define PETSC_CUPMBLAS_PREFIX         hipblas
    #define PETSC_CUPMBLAS_PREFIX_U       HIPBLAS
    #define PETSC_CUPMBLAS_FP_TYPE        PETSC_CUPMBLAS_FP_TYPE_U
    #define PETSC_CUPMBLAS_FP_INPUT_TYPE  PETSC_CUPMBLAS_FP_INPUT_TYPE_U
    #define PETSC_CUPMBLAS_FP_RETURN_TYPE PETSC_CUPMBLAS_FP_RETURN_TYPE_L
template <>
struct BlasInterfaceImpl<DeviceType::HIP> : Interface<DeviceType::HIP> {
  // typedefs
  using cupmBlasHandle_t      = cupmBlasHandleWrapper<hipblasHandle_t, 0>;
  using cupmBlasError_t       = hipblasStatus_t;
  using cupmBlasInt_t         = int; // rocblas will have its own
  using cupmBlasPointerMode_t = hipblasPointerMode_t;

  // values
  static const auto CUPMBLAS_STATUS_SUCCESS         = HIPBLAS_STATUS_SUCCESS;
  static const auto CUPMBLAS_STATUS_NOT_INITIALIZED = HIPBLAS_STATUS_NOT_INITIALIZED;
  static const auto CUPMBLAS_STATUS_ALLOC_FAILED    = HIPBLAS_STATUS_ALLOC_FAILED;
  static const auto CUPMBLAS_POINTER_MODE_HOST      = HIPBLAS_POINTER_MODE_HOST;
  static const auto CUPMBLAS_POINTER_MODE_DEVICE    = HIPBLAS_POINTER_MODE_DEVICE;
  static const auto CUPMBLAS_OP_T                   = HIPBLAS_OP_T;
  static const auto CUPMBLAS_OP_N                   = HIPBLAS_OP_N;
  static const auto CUPMBLAS_OP_C                   = HIPBLAS_OP_C;
  static const auto CUPMBLAS_FILL_MODE_LOWER        = HIPBLAS_FILL_MODE_LOWER;
  static const auto CUPMBLAS_FILL_MODE_UPPER        = HIPBLAS_FILL_MODE_UPPER;
  static const auto CUPMBLAS_SIDE_LEFT              = HIPBLAS_SIDE_LEFT;
  static const auto CUPMBLAS_DIAG_NON_UNIT          = HIPBLAS_DIAG_NON_UNIT;

  // utility functions
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Create)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Destroy)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(GetStream)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(SetStream)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(GetPointerMode)
  PETSC_CUPMBLAS_ALIAS_FUNCTION(SetPointerMode)

  // level 1 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, axpy)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, scal)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD, dot, PetscIfPetscDefined(USE_COMPLEX, dotc, dot))
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD, dotu, PetscIfPetscDefined(USE_COMPLEX, dotu, dot))
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, swap)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED, nrm2)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(IFPTYPE, amax)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED, asum)

  // level 2 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, gemv)

  // level 3 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, gemm)
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, trsm)

  // BLAS extensions
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD, geam)

  PETSC_NODISCARD static const char *cupmBlasGetErrorName(cupmBlasError_t status) noexcept { return PetscHIPBLASGetErrorName(status); }
};
    #undef PETSC_CUPMBLAS_PREFIX
    #undef PETSC_CUPMBLAS_PREFIX_U
    #undef PETSC_CUPMBLAS_FP_TYPE
    #undef PETSC_CUPMBLAS_FP_INPUT_TYPE
    #undef PETSC_CUPMBLAS_FP_RETURN_TYPE
  #endif // PetscDefined(HAVE_HIP)

  #define PETSC_CUPMBLAS_IMPL_CLASS_HEADER(T) \
    PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T); \
    /* introspection */ \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasGetErrorName; \
    /* types */ \
    using cupmBlasHandle_t      = typename ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasHandle_t; \
    using cupmBlasError_t       = typename ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasError_t; \
    using cupmBlasInt_t         = typename ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasInt_t; \
    using cupmBlasPointerMode_t = typename ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasPointerMode_t; \
    /* values */ \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_STATUS_SUCCESS; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_STATUS_NOT_INITIALIZED; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_STATUS_ALLOC_FAILED; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_POINTER_MODE_HOST; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_POINTER_MODE_DEVICE; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_OP_T; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_OP_N; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_OP_C; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_FILL_MODE_LOWER; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_FILL_MODE_UPPER; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_SIDE_LEFT; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::CUPMBLAS_DIAG_NON_UNIT; \
    /* utility functions */ \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasCreate; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasDestroy; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasGetStream; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasSetStream; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasGetPointerMode; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasSetPointerMode; \
    /* level 1 BLAS */ \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXaxpy; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXscal; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXdot; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXdotu; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXswap; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXnrm2; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXamax; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXasum; \
    /* level 2 BLAS */ \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXgemv; \
    /* level 3 BLAS */ \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXgemm; \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXtrsm; \
    /* BLAS extensions */ \
    using ::Petsc::device::cupm::impl::BlasInterfaceImpl<T>::cupmBlasXgeam

// The actual interface class
template <DeviceType T>
struct BlasInterface : BlasInterfaceImpl<T> {
  PETSC_CUPMBLAS_IMPL_CLASS_HEADER(T);

  PETSC_NODISCARD static constexpr const char *cupmBlasName() noexcept { return T == DeviceType::CUDA ? "cuBLAS" : "hipBLAS"; }

  static PetscErrorCode PetscCUPMBlasSetPointerModeFromPointer(cupmBlasHandle_t handle, const void *ptr) noexcept
  {
    auto mtype = PETSC_MEMTYPE_HOST;

    PetscFunctionBegin;
    PetscCall(PetscCUPMGetMemType(ptr, &mtype));
    PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, PetscMemTypeDevice(mtype) ? CUPMBLAS_POINTER_MODE_DEVICE : CUPMBLAS_POINTER_MODE_HOST));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode checkCupmBlasIntCast(PetscInt x) noexcept
  {
    PetscFunctionBegin;
    PetscCheck((std::is_same<PetscInt, cupmBlasInt_t>::value) || (x <= std::numeric_limits<cupmBlasInt_t>::max()), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT " is too big for %s, which may be restricted to 32 bit integers", x, cupmBlasName());
    PetscCheck(x >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Passing negative integer (%" PetscInt_FMT ") to %s routine", x, cupmBlasName());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode PetscCUPMBlasIntCast(PetscInt x, cupmBlasInt_t *y) noexcept
  {
    PetscFunctionBegin;
    *y = static_cast<cupmBlasInt_t>(x);
    PetscCall(checkCupmBlasIntCast(x));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

  #define PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(T) \
    PETSC_CUPMBLAS_IMPL_CLASS_HEADER(T); \
    using ::Petsc::device::cupm::impl::BlasInterface<T>::cupmBlasName; \
    using ::Petsc::device::cupm::impl::BlasInterface<T>::PetscCUPMBlasSetPointerModeFromPointer; \
    using ::Petsc::device::cupm::impl::BlasInterface<T>::checkCupmBlasIntCast; \
    using ::Petsc::device::cupm::impl::BlasInterface<T>::PetscCUPMBlasIntCast

  #if PetscDefined(HAVE_CUDA)
extern template struct BlasInterface<DeviceType::CUDA>;
  #endif

  #if PetscDefined(HAVE_HIP)
extern template struct BlasInterface<DeviceType::HIP>;
  #endif

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // defined(__cplusplus)

#endif // PETSCCUPMBLASINTERFACE_HPP
