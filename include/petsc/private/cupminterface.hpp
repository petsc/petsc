#ifndef PETSCCUPMINTERFACE_HPP
#define PETSCCUPMINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/traithelpers.hpp>

#if defined(__cplusplus)
#if !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#error CUPMInterface requires c++11
#endif // PetscDefined(HAVE_CXX_DIALECT_CXX11)

namespace Petsc
{

// enum describing available cupm devices, this is used as the template parameter to any
// class subclassing the CUPMInterface or using it as a member variable
enum class CUPMDeviceType : int {
  CUDA,
  HIP
};

static constexpr const char *const CUPMDeviceTypes[] = {
  "cuda",
  "hip",
  "CUPMDeviceType",
  "CUPMDeviceType::",
  nullptr
};

#if defined(CHKERRCUPM)
#  error "Invalid redefinition of CHKERRCUPM, perhaps change order of header-file includes"
#endif

// A backend agnostic CHKERRCUPM() function, this will only work inside the member
// functions of a class inheriting from CUPMInterface
#define CHKERRCUPM(cerr) do {                                           \
    cupmError_t _cerr__ = (cerr);                                       \
    if (PetscUnlikely(_cerr__ != cupmSuccess)) {                        \
      const auto name    = cupmGetErrorName(_cerr__);                   \
      const auto desc    = cupmGetErrorString(_cerr__);                 \
      const auto backend = cupmName();                                  \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",   \
               backend,static_cast<PetscErrorCode>(_cerr__),name,desc); \
    }                                                                   \
  } while (0)

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <CUPMDeviceType T> struct CUPMInterface;

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT_(prefix,original,mapped)  \
  static const auto cupm ## mapped = prefix ## original

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(prefix,original,mapped)   \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT_(prefix,original,mapped)

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE(prefix,common)          \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(prefix,common,common)

#define PETSC_CUPM_ALIAS_FUNCTION_(prefix,stem)                         \
  PETSC_ALIAS_FUNCTION(static constexpr cupm ## stem, prefix ## stem)

#define PETSC_CUPM_ALIAS_FUNCTION(prefix,stem) PETSC_CUPM_ALIAS_FUNCTION_(prefix,stem)

#if PetscDefined(HAVE_CUDA)
template <>
struct CUPMInterface<CUPMDeviceType::CUDA>
{
  static constexpr CUPMDeviceType type = CUPMDeviceType::CUDA;

  PETSC_NODISCARD static constexpr const char* cupmName() noexcept
  { return CUPMDeviceTypes[static_cast<int>(type)]; }

  // typedefs
  using cupmError_t        = cudaError_t;
  using cupmEvent_t        = cudaEvent_t;
  using cupmStream_t       = cudaStream_t;
  using cupmBlasHandle_t   = cublasHandle_t;
  using cupmBlasError_t    = cublasStatus_t;
  using cupmSolverHandle_t = cusolverDnHandle_t;
  using cupmSolverError_t  = cusolverStatus_t;
  using cupmDeviceProp_t   = cudaDeviceProp;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,ErrorNotReady);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,ErrorDeviceAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,ErrorSetOnActiveProcess);
#if PETSC_PKG_CUDA_VERSION_GE(11,1,0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,ErrorStubLibrary);
#else
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cuda,ErrorInsufficientDriver,ErrorStubLibrary);
#endif
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(cuda,MemcpyHostToDevice);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(cuda,GetErrorName);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,GetErrorString);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,GetLastError);

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(cuda,GetDeviceCount);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,GetDeviceProperties);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,GetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,SetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,GetDeviceFlags);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,SetDeviceFlags);

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(cuda,EventCreate);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,EventDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,EventRecord);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,EventSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,EventElapsedTime);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,StreamCreate);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,StreamCreateWithFlags);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,StreamDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,StreamWaitEvent);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,StreamQuery);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,StreamSynchronize);

  // general purpose
  PETSC_CUPM_ALIAS_FUNCTION(cuda,Free);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,Malloc);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(cuda,DeviceSynchronize);

  // There isn't a good way to auto-template this stuff between the cublas handle and
  // cusolver handle, not in the least because CHKERRCUBLAS and CHKERRCUSOLVER (not to
  // mention their HIP counterparts) do ~slightly~ different things. So we just overload
  // and accept the bloat
  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) noexcept
  {
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    if (handle) PetscFunctionReturn(0);
    for (int i = 0; i < 3; ++i) {
      PetscErrorCode ierr;

      cberr = cublasCreate(&handle);
      if (cberr == CUBLAS_STATUS_SUCCESS) break;
      if ((cberr != CUBLAS_STATUS_ALLOC_FAILED) && (cberr != CUBLAS_STATUS_NOT_INITIALIZED)) CHKERRCUBLAS(cberr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (PetscUnlikely(cberr != CUBLAS_STATUS_SUCCESS)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) noexcept
  {
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    if (handle) PetscFunctionReturn(0);
    for (int i = 0; i < 3; ++i) {
      PetscErrorCode ierr;

      cerr = cusolverDnCreate(&handle);
      if (cerr == CUSOLVER_STATUS_SUCCESS) break;
      if ((cerr != CUSOLVER_STATUS_NOT_INITIALIZED) && (cerr != CUSOLVER_STATUS_ALLOC_FAILED)) CHKERRCUSOLVER(cerr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (PetscUnlikely(cerr != CUSOLVER_STATUS_SUCCESS)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSolverDn");
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) noexcept
  {
    cupmStream_t    cupmStream;
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    cberr = cublasGetStream(handle,&cupmStream);CHKERRCUBLAS(cberr);
    if (cupmStream != stream) {
      cberr = cublasSetStream(handle,stream);CHKERRCUBLAS(cberr);
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) noexcept
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = cusolverDnGetStream(handle,&cupmStream);CHKERRCUSOLVER(cerr);
    if (cupmStream != stream) {
      cerr = cusolverDnSetStream(handle,stream);CHKERRCUSOLVER(cerr);
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = cublasDestroy(handle);CHKERRCUBLAS(cberr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr   = cusolverDnDestroy(handle);CHKERRCUSOLVER(cerr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }
};
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
template <>
struct CUPMInterface<CUPMDeviceType::HIP>
{
  static constexpr CUPMDeviceType type = CUPMDeviceType::HIP;

  PETSC_NODISCARD static constexpr const char* cupmName() noexcept
  { return CUPMDeviceTypes[static_cast<int>(type)];}

  // typedefs
  using cupmError_t        = hipError_t;
  using cupmEvent_t        = hipEvent_t;
  using cupmStream_t       = hipStream_t;
  using cupmBlasHandle_t   = hipblasHandle_t;
  using cupmBlasError_t    = hipblasStatus_t;
  using cupmSolverHandle_t = hipsolverHandle_t;
  using cupmSolverError_t  = hipsolverStatus_t;
  using cupmDeviceProp_t   = hipDeviceProp_t;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(hip,Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(hip,ErrorNotReady);
  // see https://github.com/ROCm-Developer-Tools/HIP/blob/develop/bin/hipify-perl
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(hip,ErrorContextAlreadyInUse,ErrorDeviceAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(hip,ErrorSetOnActiveProcess);
  // as of HIP v4.2 cudaErrorStubLibrary has no HIP equivalent
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(hip,ErrorInsufficientDriver,ErrorStubLibrary);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(hip,StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(hip,DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(hip,MemcpyHostToDevice);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(hip,GetErrorName);
  PETSC_CUPM_ALIAS_FUNCTION(hip,GetErrorString);
  PETSC_CUPM_ALIAS_FUNCTION(hip,GetLastError);

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(hip,GetDeviceCount);
  PETSC_CUPM_ALIAS_FUNCTION(hip,GetDeviceProperties);
  PETSC_CUPM_ALIAS_FUNCTION(hip,GetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(hip,SetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(hip,GetDeviceFlags);
  PETSC_CUPM_ALIAS_FUNCTION(hip,SetDeviceFlags);

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(hip,EventCreate);
  PETSC_CUPM_ALIAS_FUNCTION(hip,EventDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(hip,EventRecord);
  PETSC_CUPM_ALIAS_FUNCTION(hip,EventSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(hip,EventElapsedTime);
  PETSC_CUPM_ALIAS_FUNCTION(hip,StreamCreate);
  PETSC_CUPM_ALIAS_FUNCTION(hip,StreamCreateWithFlags);
  PETSC_CUPM_ALIAS_FUNCTION(hip,StreamDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(hip,StreamWaitEvent);
  PETSC_CUPM_ALIAS_FUNCTION(hip,StreamQuery);
  PETSC_CUPM_ALIAS_FUNCTION(hip,StreamSynchronize);

  // general purpose
  PETSC_CUPM_ALIAS_FUNCTION(hip,Free);
  PETSC_CUPM_ALIAS_FUNCTION(hip,Malloc);
  PETSC_CUPM_ALIAS_FUNCTION(hip,Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(hip,DeviceSynchronize);

  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (!handle) {cupmBlasError_t cberr = hipblasCreate(&handle);CHKERRHIPBLAS(cberr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (!handle) {cupmSolverError_t cerr = hipsolverCreate(&handle);CHKERRHIPSOLVER(cerr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) noexcept
  {
    cupmStream_t    cupmStream;
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    cberr = hipblasGetStream(handle,&cupmStream);CHKERRHIPBLAS(cberr);
    if (cupmStream != stream) {cberr = hipblasSetStream(handle,stream);CHKERRHIPBLAS(cberr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) noexcept
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = hipsolverGetStream(handle,&cupmStream);CHKERRHIPSOLVER(cerr);
    if (cupmStream != stream) {cerr = hipsolverSetStream(handle,stream);CHKERRHIPSOLVER(cerr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = hipblasDestroy(handle);CHKERRHIPBLAS(cberr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr   = hipsolverDestroy(handle);CHKERRHIPSOLVER(cerr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }
};
#endif // PetscDefined(HAVE_HIP)

#undef PETSC_CUPM_ALIAS_INTEGRAL_VALUE
#undef PETSC_CUPM_ALIAS_FUNCTION

} // namespace Petsc

// shorthand for bringing all of the typedefs from the base CUPMInterface class into your own,
// it's annoying that c++ doesn't have a way to do this automatically
#define PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING_(base_name_,Tp_)    \
  using base_name_ = CUPMInterface<Tp_>;                                \
  /* introspective typedefs */                                          \
  using base_name_::type;                                               \
  /* types */                                                           \
  using typename base_name_::cupmError_t;                               \
  using typename base_name_::cupmEvent_t;                               \
  using typename base_name_::cupmStream_t;                              \
  using typename base_name_::cupmBlasError_t;                           \
  using typename base_name_::cupmSolverError_t;                         \
  using typename base_name_::cupmBlasHandle_t;                          \
  using typename base_name_::cupmSolverHandle_t;                        \
  using typename base_name_::cupmDeviceProp_t;                          \
  /* variables */                                                       \
  using base_name_::cupmSuccess;                                        \
  using base_name_::cupmErrorNotReady;                                  \
  using base_name_::cupmErrorStubLibrary;                               \
  using base_name_::cupmErrorDeviceAlreadyInUse;                        \
  using base_name_::cupmErrorSetOnActiveProcess;                        \
  using base_name_::cupmStreamNonBlocking;                              \
  using base_name_::cupmDeviceMapHost;                                  \
  using base_name_::cupmMemcpyHostToDevice;                             \
  /* functions */                                                       \
  using base_name_::cupmName;                                           \
  using base_name_::cupmGetErrorName;                                   \
  using base_name_::cupmGetErrorString;                                 \
  using base_name_::cupmGetLastError;                                   \
  using base_name_::cupmGetDeviceCount;                                 \
  using base_name_::cupmGetDeviceProperties;                            \
  using base_name_::cupmGetDevice;                                      \
  using base_name_::cupmSetDevice;                                      \
  using base_name_::cupmGetDeviceFlags;                                 \
  using base_name_::cupmSetDeviceFlags;                                 \
  using base_name_::cupmEventCreate;                                    \
  using base_name_::cupmEventDestroy;                                   \
  using base_name_::cupmEventRecord;                                    \
  using base_name_::cupmEventSynchronize;                               \
  using base_name_::cupmEventElapsedTime;                               \
  using base_name_::cupmStreamCreate;                                   \
  using base_name_::cupmStreamCreateWithFlags;                          \
  using base_name_::cupmStreamDestroy;                                  \
  using base_name_::cupmStreamWaitEvent;                                \
  using base_name_::cupmStreamQuery;                                    \
  using base_name_::cupmStreamSynchronize;                              \
  using base_name_::cupmFree;                                           \
  using base_name_::cupmMalloc;                                         \
  using base_name_::cupmMemcpy;                                         \
  using base_name_::cupmDeviceSynchronize;

// allow any macros to expand in case someone needs it
#define PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(base_name_,Tp_)     \
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING_(base_name_,Tp_)

#endif /* __cplusplus */

#endif /* PETSC_CUPMTRAITS_HPP */
