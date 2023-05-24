#ifndef PETSCCUPMINTERFACE_HPP
#define PETSCCUPMINTERFACE_HPP

#if defined(__cplusplus)
  #include <petsc/private/cpputil.hpp>
  #include <petsc/private/petscadvancedmacros.h>
  #include <petscdevice_cupm.h>

  #include <array>

namespace Petsc
{

namespace device
{

namespace cupm
{

// enum describing available cupm devices, this is used as the template parameter to any
// class subclassing the Interface or using it as a member variable
enum class DeviceType : int {
  CUDA,
  HIP
};

// clang-format off
static constexpr std::array<const char *const, 5> DeviceTypes = {
  "cuda",
  "hip",
  "Petsc::Device::CUPM::DeviceType",
  "Petsc::Device::CUPM::DeviceType::",
  nullptr
};
// clang-format on

namespace impl
{

  // A backend agnostic PetscCallCUPM() function, this will only work inside the member
  // functions of a class inheriting from CUPM::Interface. Thanks to __VA_ARGS__ templated
  // functions can also be wrapped inline:
  //
  // PetscCallCUPM(foo<int,char,bool>());
  #define PetscCallCUPM(...) \
    do { \
      const cupmError_t cerr_p_ = __VA_ARGS__; \
      PetscCheck(cerr_p_ == cupmSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "%s error %d (%s) : %s", cupmName(), static_cast<PetscErrorCode>(cerr_p_), cupmGetErrorName(cerr_p_), cupmGetErrorString(cerr_p_)); \
    } while (0)

  #define PetscCallCUPMAbort(comm_, ...) \
    do { \
      const cupmError_t cerr_abort_p_ = __VA_ARGS__; \
      PetscCheckAbort(cerr_abort_p_ == cupmSuccess, comm_, PETSC_ERR_GPU, "%s error %d (%s) : %s", cupmName(), static_cast<PetscErrorCode>(cerr_abort_p_), cupmGetErrorName(cerr_abort_p_), cupmGetErrorString(cerr_abort_p_)); \
    } while (0)

  // PETSC_CUPM_ALIAS_FUNCTION() - declaration to alias a cuda/hip function
  //
  // input params:
  // our_name   - the name of the alias
  // their_name - the name of the function being aliased
  //
  // notes:
  // see PETSC_ALIAS_FUNCTION() for the exact nature of the expansion
  //
  // example usage:
  // PETSC_CUPM_ALIAS_FUNCTION(cupmMalloc, cudaMalloc) ->
  // template <typename... T>
  // static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
  // {
  //   return cudaMalloc(std::forward<T>(args)...);
  // }
  //
  // PETSC_CUPM_ALIAS_FUNCTION(cupmMalloc, hipMalloc) ->
  // template <typename... T>
  // static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
  // {
  //   return hipMalloc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPM_ALIAS_FUNCTION(our_name, their_name) PETSC_ALIAS_FUNCTION(static our_name, their_name)

  // PETSC_CUPM_ALIAS_FUNCTION_GOBBLE() - declaration to alias a cuda/hip function but
  // discard the last N arguments
  //
  // input params:
  // our_name   - the name of the alias
  // their_name - the name of the function being aliased
  // N          - integer constant [0, INT_MAX) dictating how many arguments to chop off the end
  //
  // notes:
  // see PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS() for the exact nature of the expansion
  //
  // example use:
  // PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(cupmMallocAsync, cudaMalloc, 1) ->
  // template <typename... T, typename Tend>
  // static constexpr auto cupmMallocAsync(T&&... args, Tend argend) *noexcept and trailing
  // return type deduction*
  // {
  //   (void)argend;
  //   return cudaMalloc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(our_name, their_name, N) PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(static our_name, their_name, N)

// Base class that holds functions and variables that don't require CUDA or HIP to be present
// on the system
template <DeviceType T>
struct InterfaceBase {
  static const DeviceType type = T;

  PETSC_NODISCARD static constexpr const char *cupmName() noexcept
  {
    static_assert(util::to_underlying(DeviceType::CUDA) == 0, "");
    static_assert(util::to_underlying(DeviceType::HIP) == 1, "");
    return std::get<util::to_underlying(T)>(DeviceTypes);
  }

  PETSC_NODISCARD static constexpr const char *cupmNAME() noexcept { return T == DeviceType::CUDA ? "CUDA" : "HIP"; }

  PETSC_NODISCARD static constexpr PetscDeviceType PETSC_DEVICE_CUPM() noexcept { return T == DeviceType::CUDA ? PETSC_DEVICE_CUDA : PETSC_DEVICE_HIP; }

  PETSC_NODISCARD static constexpr PetscMemType PETSC_MEMTYPE_CUPM() noexcept { return T == DeviceType::CUDA ? PETSC_MEMTYPE_CUDA : PETSC_MEMTYPE_HIP; }
};

// declare the base class static member variables
template <DeviceType T>
const DeviceType InterfaceBase<T>::type;

  #define PETSC_CUPM_BASE_CLASS_HEADER(T) \
    using ::Petsc::device::cupm::impl::InterfaceBase<T>::type; \
    using ::Petsc::device::cupm::impl::InterfaceBase<T>::cupmName; \
    using ::Petsc::device::cupm::impl::InterfaceBase<T>::cupmNAME; \
    using ::Petsc::device::cupm::impl::InterfaceBase<T>::PETSC_DEVICE_CUPM; \
    using ::Petsc::device::cupm::impl::InterfaceBase<T>::PETSC_MEMTYPE_CUPM

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <DeviceType>
struct InterfaceImpl;

  #if PetscDefined(HAVE_CUDA)
template <>
struct InterfaceImpl<DeviceType::CUDA> : InterfaceBase<DeviceType::CUDA> {
  PETSC_CUPM_BASE_CLASS_HEADER(DeviceType::CUDA);

  // typedefs
  using cupmError_t             = cudaError_t;
  using cupmEvent_t             = cudaEvent_t;
  using cupmStream_t            = cudaStream_t;
  using cupmDeviceProp_t        = cudaDeviceProp;
  using cupmMemcpyKind_t        = cudaMemcpyKind;
  using cupmComplex_t           = util::conditional_t<PetscDefined(USE_REAL_SINGLE), cuComplex, cuDoubleComplex>;
  using cupmPointerAttributes_t = cudaPointerAttributes;
  using cupmMemoryType_t        = enum cudaMemoryType;
  using cupmDim3                = dim3;
  using cupmHostFn_t            = cudaHostFn_t;
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  using cupmMemPool_t   = cudaMemPool_t;
  using cupmMemPoolAttr = cudaMemPoolAttr;
    #else
  using cupmMemPool_t   = void *;
  using cupmMemPoolAttr = unsigned int;
    #endif

  // values
  static const auto cupmSuccess                 = cudaSuccess;
  static const auto cupmErrorNotReady           = cudaErrorNotReady;
  static const auto cupmErrorDeviceAlreadyInUse = cudaErrorDeviceAlreadyInUse;
  static const auto cupmErrorSetOnActiveProcess = cudaErrorSetOnActiveProcess;
  static const auto cupmErrorStubLibrary =
    #if PETSC_PKG_CUDA_VERSION_GE(11, 1, 0)
    cudaErrorStubLibrary;
    #else
    cudaErrorInsufficientDriver;
    #endif

  static const auto cupmErrorNoDevice          = cudaErrorNoDevice;
  static const auto cupmStreamDefault          = cudaStreamDefault;
  static const auto cupmStreamNonBlocking      = cudaStreamNonBlocking;
  static const auto cupmDeviceMapHost          = cudaDeviceMapHost;
  static const auto cupmMemcpyHostToDevice     = cudaMemcpyHostToDevice;
  static const auto cupmMemcpyDeviceToHost     = cudaMemcpyDeviceToHost;
  static const auto cupmMemcpyDeviceToDevice   = cudaMemcpyDeviceToDevice;
  static const auto cupmMemcpyHostToHost       = cudaMemcpyHostToHost;
  static const auto cupmMemcpyDefault          = cudaMemcpyDefault;
  static const auto cupmMemoryTypeHost         = cudaMemoryTypeHost;
  static const auto cupmMemoryTypeDevice       = cudaMemoryTypeDevice;
  static const auto cupmMemoryTypeManaged      = cudaMemoryTypeManaged;
  static const auto cupmEventDisableTiming     = cudaEventDisableTiming;
  static const auto cupmHostAllocDefault       = cudaHostAllocDefault;
  static const auto cupmHostAllocWriteCombined = cudaHostAllocWriteCombined;
  static const auto cupmMemPoolAttrReleaseThreshold =
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
    cudaMemPoolAttrReleaseThreshold;
    #else
    cupmMemPoolAttr{0};
    #endif

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetErrorName, cudaGetErrorName)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetErrorString, cudaGetErrorString)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetLastError, cudaGetLastError)

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDeviceCount, cudaGetDeviceCount)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDeviceProperties, cudaGetDeviceProperties)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDevice, cudaGetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSetDevice, cudaSetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDeviceFlags, cudaGetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSetDeviceFlags, cudaSetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmPointerGetAttributes, cudaPointerGetAttributes)
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(cupmDeviceGetMemPool, cudaDeviceGetMemPool)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemPoolSetAttribute, cudaMemPoolSetAttribute)
    #else
  PETSC_NODISCARD static cupmError_t cupmDeviceGetMemPool(cupmMemPool_t *pool, int) noexcept
  {
    *pool = nullptr;
    return cupmSuccess;
  }

  PETSC_NODISCARD static cupmError_t cupmMemPoolSetAttribute(cupmMemPool_t, cupmMemPoolAttr, void *) noexcept { return cupmSuccess; }
    #endif
  // CUDA has no cudaInit() to match hipInit()
  PETSC_NODISCARD static cupmError_t cupmInit(unsigned int) noexcept { return cudaFree(nullptr); }

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventCreate, cudaEventCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventCreateWithFlags, cudaEventCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventDestroy, cudaEventDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventRecord, cudaEventRecord)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventSynchronize, cudaEventSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventElapsedTime, cudaEventElapsedTime)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventQuery, cudaEventQuery)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamCreate, cudaStreamCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamCreateWithFlags, cudaStreamCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamGetFlags, cudaStreamGetFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamDestroy, cudaStreamDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamWaitEvent, cudaStreamWaitEvent)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamQuery, cudaStreamQuery)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamSynchronize, cudaStreamSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(cupmDeviceSynchronize, cudaDeviceSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetSymbolAddress, cudaGetSymbolAddress)

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(cupmFree, cudaFree)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMalloc, cudaMalloc)
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(cupmFreeAsync, cudaFreeAsync)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMallocAsync, cudaMallocAsync)
    #else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmFreeAsync, cudaFree, 1)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmMallocAsync, cudaMalloc, 1)
    #endif
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemcpy, cudaMemcpy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemcpyAsync, cudaMemcpyAsync)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMallocHost, cudaMallocHost)
  PETSC_CUPM_ALIAS_FUNCTION(cupmFreeHost, cudaFreeHost)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemset, cudaMemset)
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemsetAsync, cudaMemsetAsync)
    #else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmMemsetAsync, cudaMemset, 1)
    #endif
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemcpy2D, cudaMemcpy2D)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmMemcpy2DAsync, cudaMemcpy2DAsync, 1)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemset2D, cudaMemset2D)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmMemset2DAsync, cudaMemset2DAsync, 1)

  // launch control
  PETSC_CUPM_ALIAS_FUNCTION(cupmLaunchHostFunc, cudaLaunchHostFunc)
  template <typename FunctionT, typename... KernelArgsT>
  PETSC_NODISCARD static cudaError_t cupmLaunchKernel(FunctionT &&func, dim3 gridDim, dim3 blockDim, std::size_t sharedMem, cudaStream_t stream, KernelArgsT &&...kernelArgs) noexcept
  {
    static_assert(!std::is_pointer<FunctionT>::value, "kernel function must not be passed by pointer");

    void *args[] = {(void *)&kernelArgs...};
    return cudaLaunchKernel<util::remove_reference_t<FunctionT>>(std::addressof(func), std::move(gridDim), std::move(blockDim), args, sharedMem, std::move(stream));
  }
};
  #endif // PetscDefined(HAVE_CUDA)

  #if PetscDefined(HAVE_HIP)
template <>
struct InterfaceImpl<DeviceType::HIP> : InterfaceBase<DeviceType::HIP> {
  PETSC_CUPM_BASE_CLASS_HEADER(DeviceType::HIP);

  // typedefs
  using cupmError_t             = hipError_t;
  using cupmEvent_t             = hipEvent_t;
  using cupmStream_t            = hipStream_t;
  using cupmDeviceProp_t        = hipDeviceProp_t;
  using cupmMemcpyKind_t        = hipMemcpyKind;
  using cupmComplex_t           = util::conditional_t<PetscDefined(USE_REAL_SINGLE), hipComplex, hipDoubleComplex>;
  using cupmPointerAttributes_t = hipPointerAttribute_t;
  using cupmMemoryType_t        = enum hipMemoryType;
  using cupmDim3                = dim3;
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  using cupmHostFn_t    = hipHostFn_t;
  using cupmMemPool_t   = hipMemPool_t;
  using cupmMemPoolAttr = hipMemPoolAttr;
    #else
  using cupmHostFn_t    = void (*)(void *);
  using cupmMemPool_t   = void *;
  using cupmMemPoolAttr = unsigned int;
    #endif

  // values
  static const auto cupmSuccess       = hipSuccess;
  static const auto cupmErrorNotReady = hipErrorNotReady;
  // see https://github.com/ROCm-Developer-Tools/HIP/blob/develop/bin/hipify-perl
  static const auto cupmErrorDeviceAlreadyInUse = hipErrorContextAlreadyInUse;
  static const auto cupmErrorSetOnActiveProcess = hipErrorSetOnActiveProcess;
  // as of HIP v4.2 cudaErrorStubLibrary has no HIP equivalent
  static const auto cupmErrorStubLibrary     = hipErrorInsufficientDriver;
  static const auto cupmErrorNoDevice        = hipErrorNoDevice;
  static const auto cupmStreamDefault        = hipStreamDefault;
  static const auto cupmStreamNonBlocking    = hipStreamNonBlocking;
  static const auto cupmDeviceMapHost        = hipDeviceMapHost;
  static const auto cupmMemcpyHostToDevice   = hipMemcpyHostToDevice;
  static const auto cupmMemcpyDeviceToHost   = hipMemcpyDeviceToHost;
  static const auto cupmMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
  static const auto cupmMemcpyHostToHost     = hipMemcpyHostToHost;
  static const auto cupmMemcpyDefault        = hipMemcpyDefault;
  static const auto cupmMemoryTypeHost       = hipMemoryTypeHost;
  static const auto cupmMemoryTypeDevice     = hipMemoryTypeDevice;
  // see
  // https://github.com/ROCm-Developer-Tools/HIP/blob/develop/include/hip/hip_runtime_api.h#L156
  static const auto cupmMemoryTypeManaged      = hipMemoryTypeUnified;
  static const auto cupmEventDisableTiming     = hipEventDisableTiming;
  static const auto cupmHostAllocDefault       = hipHostMallocDefault;
  static const auto cupmHostAllocWriteCombined = hipHostMallocWriteCombined;
  static const auto cupmMemPoolAttrReleaseThreshold =
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
    hipMemPoolAttrReleaseThreshold;
    #else
    cupmMemPoolAttr{0};
    #endif

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetErrorName, hipGetErrorName)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetErrorString, hipGetErrorString)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetLastError, hipGetLastError)

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDeviceCount, hipGetDeviceCount)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDeviceProperties, hipGetDeviceProperties)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDevice, hipGetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSetDevice, hipSetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetDeviceFlags, hipGetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmSetDeviceFlags, hipSetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmPointerGetAttributes, hipPointerGetAttributes)
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(cupmDeviceGetMemPool, hipDeviceGetMemPool)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemPoolSetAttribute, hipMemPoolSetAttribute)
    #else
  PETSC_NODISCARD static cupmError_t cupmDeviceGetMemPool(cupmMemPool_t *pool, int) noexcept
  {
    *pool = nullptr;
    return cupmSuccess;
  }

  PETSC_NODISCARD static cupmError_t cupmMemPoolSetAttribute(cupmMemPool_t, cupmMemPoolAttr, void *) noexcept { return cupmSuccess; }
    #endif
  PETSC_CUPM_ALIAS_FUNCTION(cupmInit, hipInit)

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventCreate, hipEventCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventCreateWithFlags, hipEventCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventDestroy, hipEventDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventRecord, hipEventRecord)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventSynchronize, hipEventSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventElapsedTime, hipEventElapsedTime)
  PETSC_CUPM_ALIAS_FUNCTION(cupmEventQuery, hipEventQuery)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamCreate, hipStreamCreate)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamCreateWithFlags, hipStreamCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamGetFlags, hipStreamGetFlags)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamDestroy, hipStreamDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamWaitEvent, hipStreamWaitEvent)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamQuery, hipStreamQuery)
  PETSC_CUPM_ALIAS_FUNCTION(cupmStreamSynchronize, hipStreamSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(cupmDeviceSynchronize, hipDeviceSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(cupmGetSymbolAddress, hipGetSymbolAddress)

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(cupmFree, hipFree)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMalloc, hipMalloc)
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMallocAsync, hipMallocAsync)
  PETSC_CUPM_ALIAS_FUNCTION(cupmFreeAsync, hipFreeAsync)
    #else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmMallocAsync, hipMalloc, 1)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmFreeAsync, hipFree, 1)
    #endif
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemcpy, hipMemcpy)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemcpyAsync, hipMemcpyAsync)
  // hipMallocHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION(cupmMallocHost, hipHostMalloc)
  // hipFreeHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION(cupmFreeHost, hipHostFree)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemset, hipMemset)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemsetAsync, hipMemsetAsync)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemcpy2D, hipMemcpy2D)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmMemcpy2DAsync, hipMemcpy2DAsync, 1)
  PETSC_CUPM_ALIAS_FUNCTION(cupmMemset2D, hipMemset2D)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE(cupmMemset2DAsync, hipMemset2DAsync, 1)

      // launch control
      // HIP appears to only have hipLaunchHostFunc from 5.2.0 onwards
      // https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md#7-execution-control=
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(cupmLaunchHostFunc, hipLaunchHostFunc)
    #else
  PETSC_NODISCARD static hipError_t cupmLaunchHostFunc(hipStream_t stream, cupmHostFn_t fn, void *ctx) noexcept
  {
    // the only correct way to spoof this function is to do it synchronously...
    auto herr = hipStreamSynchronize(stream);
    if (PetscUnlikely(herr != hipSuccess)) return herr;
    fn(ctx);
    return herr;
  }
    #endif

  template <typename FunctionT, typename... KernelArgsT>
  PETSC_NODISCARD static hipError_t cupmLaunchKernel(FunctionT &&func, dim3 gridDim, dim3 blockDim, std::size_t sharedMem, hipStream_t stream, KernelArgsT &&...kernelArgs) noexcept
  {
    void *args[] = {(void *)&kernelArgs...};
    return hipLaunchKernel((void *)func, std::move(gridDim), std::move(blockDim), args, sharedMem, std::move(stream));
  }
};
  #endif // PetscDefined(HAVE_HIP)

  // shorthand for bringing all of the typedefs from the base Interface class into your own,
  // it's annoying that c++ doesn't have a way to do this automatically
  #define PETSC_CUPM_IMPL_CLASS_HEADER(T) \
    PETSC_CUPM_BASE_CLASS_HEADER(T); \
    /* types */ \
    using cupmComplex_t           = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmComplex_t; \
    using cupmError_t             = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmError_t; \
    using cupmEvent_t             = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEvent_t; \
    using cupmStream_t            = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStream_t; \
    using cupmDeviceProp_t        = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmDeviceProp_t; \
    using cupmMemcpyKind_t        = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpyKind_t; \
    using cupmPointerAttributes_t = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmPointerAttributes_t; \
    using cupmMemoryType_t        = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemoryType_t; \
    using cupmDim3                = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmDim3; \
    using cupmMemPool_t           = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemPool_t; \
    using cupmMemPoolAttr         = typename ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemPoolAttr; \
    /* variables */ \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmSuccess; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmErrorNotReady; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmErrorDeviceAlreadyInUse; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmErrorSetOnActiveProcess; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmErrorStubLibrary; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmErrorNoDevice; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamDefault; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamNonBlocking; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmDeviceMapHost; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpyHostToDevice; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpyDeviceToHost; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpyDeviceToDevice; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpyHostToHost; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpyDefault; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemoryTypeHost; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemoryTypeDevice; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemoryTypeManaged; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventDisableTiming; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmHostAllocDefault; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmHostAllocWriteCombined; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemPoolAttrReleaseThreshold; \
    /* functions */ \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetErrorName; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetErrorString; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetLastError; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetDeviceCount; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetDeviceProperties; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetDevice; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmSetDevice; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetDeviceFlags; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmSetDeviceFlags; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmPointerGetAttributes; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmDeviceGetMemPool; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemPoolSetAttribute; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmInit; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventCreate; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventCreateWithFlags; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventDestroy; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventRecord; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventSynchronize; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventElapsedTime; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmEventQuery; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamCreate; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamCreateWithFlags; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamGetFlags; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamDestroy; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamWaitEvent; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamQuery; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmStreamSynchronize; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmDeviceSynchronize; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmGetSymbolAddress; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMalloc; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMallocAsync; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpy; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpyAsync; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMallocHost; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemset; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemsetAsync; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpy2D; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemcpy2DAsync; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemset2D; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmMemset2DAsync; \
    using ::Petsc::device::cupm::impl::InterfaceImpl<T>::cupmLaunchHostFunc

// The actual interface class
template <DeviceType T>
struct Interface : InterfaceImpl<T> {
private:
  using interface_type = InterfaceImpl<T>;

public:
  PETSC_CUPM_IMPL_CLASS_HEADER(T);

  using cupmReal_t   = util::conditional_t<PetscDefined(USE_REAL_SINGLE), float, double>;
  using cupmScalar_t = util::conditional_t<PetscDefined(USE_COMPLEX), cupmComplex_t, cupmReal_t>;

  PETSC_NODISCARD static constexpr cupmScalar_t cupmScalarCast(PetscScalar s) noexcept
  {
  #if PetscDefined(USE_COMPLEX)
    return cupmComplex_t{PetscRealPart(s), PetscImaginaryPart(s)};
  #else
    return static_cast<cupmScalar_t>(s);
  #endif
  }

  PETSC_NODISCARD static constexpr const cupmScalar_t *cupmScalarPtrCast(const PetscScalar *s) noexcept { return reinterpret_cast<const cupmScalar_t *>(s); }

  PETSC_NODISCARD static constexpr cupmScalar_t *cupmScalarPtrCast(PetscScalar *s) noexcept { return reinterpret_cast<cupmScalar_t *>(s); }

  PETSC_NODISCARD static constexpr const cupmReal_t *cupmRealPtrCast(const PetscReal *s) noexcept { return reinterpret_cast<const cupmReal_t *>(s); }

  PETSC_NODISCARD static constexpr cupmReal_t *cupmRealPtrCast(PetscReal *s) noexcept { return reinterpret_cast<cupmReal_t *>(s); }

  #if !defined(PETSC_PKG_CUDA_VERSION_GE)
    #define PETSC_PKG_CUDA_VERSION_GE(...) 0
    #define CUPM_DEFINED_PETSC_PKG_CUDA_VERSION_GE
  #endif
  static PetscErrorCode PetscCUPMGetMemType(const void *data, PetscMemType *type, PetscBool *registered = nullptr, PetscBool *managed = nullptr) noexcept
  {
    cupmPointerAttributes_t attr;
    cupmError_t             cerr;

    PetscFunctionBegin;
    if (type) PetscValidPointer(type, 2);
    if (registered) {
      PetscValidBoolPointer(registered, 3);
      *registered = PETSC_FALSE;
    }
    if (managed) {
      PetscValidBoolPointer(managed, 4);
      *managed = PETSC_FALSE;
    }
    // Do not check error, instead reset it via GetLastError() since before CUDA 11.0, passing
    // a host pointer returns cudaErrorInvalidValue
    cerr = cupmPointerGetAttributes(&attr, data);
    cerr = cupmGetLastError();
      // HIP seems to always have used memoryType though
  #if (defined(CUDART_VERSION) && (CUDART_VERSION < 10000)) || defined(__HIP_PLATFORM_HCC__)
    const auto mtype = attr.memoryType;
    if (managed) *managed = static_cast<PetscBool>((cerr == cupmSuccess) && attr.isManaged);
  #else
    if (PETSC_PKG_CUDA_VERSION_GE(11, 0, 0) && (T == DeviceType::CUDA)) PetscCallCUPM(cerr);
    const auto mtype = attr.type;
    if (managed) *managed = static_cast<PetscBool>(mtype == cupmMemoryTypeManaged);
  #endif // CUDART_VERSION && CUDART_VERSION < 10000 || __HIP_PLATFORM_HCC__
    if (type) *type = ((cerr == cupmSuccess) && (mtype == cupmMemoryTypeDevice)) ? PETSC_MEMTYPE_CUPM() : PETSC_MEMTYPE_HOST;
    if (registered && (cerr == cupmSuccess) && (mtype == cupmMemoryTypeHost)) *registered = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  #if defined(CUPM_DEFINED_PETSC_PKG_CUDA_VERSION_GE)
    #undef PETSC_PKG_CUDA_VERSION_GE
  #endif

  PETSC_NODISCARD static PETSC_CONSTEXPR_14 cupmMemcpyKind_t PetscDeviceCopyModeToCUPMMemcpyKind(PetscDeviceCopyMode mode) noexcept
  {
    switch (mode) {
    case PETSC_DEVICE_COPY_HTOH:
      return cupmMemcpyHostToHost;
    case PETSC_DEVICE_COPY_HTOD:
      return cupmMemcpyHostToDevice;
    case PETSC_DEVICE_COPY_DTOD:
      return cupmMemcpyDeviceToDevice;
    case PETSC_DEVICE_COPY_DTOH:
      return cupmMemcpyDeviceToHost;
    case PETSC_DEVICE_COPY_AUTO:
      return cupmMemcpyDefault;
    }
    PetscUnreachable();
    return cupmMemcpyDefault;
  }

  // these change what the arguments mean, so need to namespace these
  template <typename M>
  static PetscErrorCode PetscCUPMMallocAsync(M **ptr, std::size_t n, cupmStream_t stream = nullptr) noexcept
  {
    static_assert(!std::is_void<M>::value, "");

    PetscFunctionBegin;
    PetscValidPointer(ptr, 1);
    *ptr = nullptr;
    if (n) {
      const auto bytes = n * sizeof(M);
      // https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/
      //
      // TLD;DR: cudaMallocAsync() does not work with NVIDIA GPUDirect which OPENMPI uses to
      // underpin its cuda-aware MPI implementation, so we cannot just async allocate
      // blindly...
      if (stream) {
        PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(ptr), bytes, stream));
      } else {
        PetscCallCUPM(cupmMalloc(reinterpret_cast<void **>(ptr), bytes));
      }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename M>
  static PetscErrorCode PetscCUPMMalloc(M **ptr, std::size_t n) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMallocAsync(ptr, n));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename M>
  static PetscErrorCode PetscCUPMMallocHost(M **ptr, std::size_t n, unsigned int flags = cupmHostAllocDefault) noexcept
  {
    static_assert(!std::is_void<M>::value, "");

    PetscFunctionBegin;
    PetscValidPointer(ptr, 1);
    *ptr = nullptr;
    if (n) PetscCallCUPM(cupmMallocHost(reinterpret_cast<void **>(ptr), n * sizeof(M), flags));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename D>
  static PetscErrorCode PetscCUPMMemcpyAsync(D *dest, const util::type_identity_t<D> *src, std::size_t n, cupmMemcpyKind_t kind, cupmStream_t stream = nullptr, bool use_async = false) noexcept
  {
    static_assert(!std::is_void<D>::value, "");
    const auto size = n * sizeof(D);

    PetscFunctionBegin;
    if (PetscUnlikely(!n)) PetscFunctionReturn(PETSC_SUCCESS);
    // cannot dereference (i.e. cannot call PetscValidPointer() here)
    PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
    PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
    // do early return after nullptr check since we need to check that they are not both nullptrs
    if (PetscUnlikely(dest == src)) PetscFunctionReturn(PETSC_SUCCESS);
    if (kind == cupmMemcpyHostToHost) {
      // If we are HTOH it is cheaper to check if the stream is idle and do a basic mempcy()
      // than it is to just call the vendor functions. This assumes of course that the stream
      // accounts for both memory regions being "idle"
      if (cupmStreamQuery(stream) == cupmSuccess) {
        PetscCall(PetscMemcpy(dest, src, size));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      // need to clear the potential cupmErrorNotReady generated by query above...
      auto cerr = cupmGetLastError();

      if (PetscUnlikely(cerr != cupmErrorNotReady)) PetscCallCUPM(cerr);
    }
    if (use_async || stream || (kind != cupmMemcpyDeviceToHost)) {
      PetscCallCUPM(cupmMemcpyAsync(dest, src, size, kind, stream));
    } else {
      PetscCallCUPM(cupmMemcpy(dest, src, size, kind));
    }
    PetscCall(PetscLogCUPMMemcpyTransfer(kind, size));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename D>
  static PetscErrorCode PetscCUPMMemcpy(D *dest, const util::type_identity_t<D> *src, std::size_t n, cupmMemcpyKind_t kind) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemcpyAsync(dest, src, n, kind));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename D>
  static PetscErrorCode PetscCUPMMemcpy2DAsync(D *dest, std::size_t dest_pitch, const util::type_identity_t<D> *src, std::size_t src_pitch, std::size_t width, std::size_t height, cupmMemcpyKind_t kind, cupmStream_t stream = nullptr)
  {
    static_assert(!std::is_void<D>::value, "");
    const auto dest_pitch_bytes = dest_pitch * sizeof(D);
    const auto src_pitch_bytes  = src_pitch * sizeof(D);
    const auto width_bytes      = width * sizeof(D);
    const auto size             = height * width_bytes;

    PetscFunctionBegin;
    if (PetscUnlikely(!size)) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
    PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
    if (stream || (kind != cupmMemcpyDeviceToHost)) {
      PetscCallCUPM(cupmMemcpy2DAsync(dest, dest_pitch_bytes, src, src_pitch_bytes, width_bytes, height, kind, stream));
    } else {
      PetscCallCUPM(cupmMemcpy2D(dest, dest_pitch_bytes, src, src_pitch_bytes, width_bytes, height, kind));
    }
    PetscCall(PetscLogCUPMMemcpyTransfer(kind, size));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename D>
  static PetscErrorCode PetscCUPMMemcpy2D(D *dest, std::size_t dest_pitch, const util::type_identity_t<D> *src, std::size_t src_pitch, std::size_t width, std::size_t height, cupmMemcpyKind_t kind)
  {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemcpy2DAsync(dest, dest_pitch, src, src_pitch, width, height, kind));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename M>
  static PetscErrorCode PetscCUPMMemsetAsync(M *ptr, int value, std::size_t n, cupmStream_t stream = nullptr, bool use_async = false) noexcept
  {
    static_assert(!std::is_void<M>::value, "");

    PetscFunctionBegin;
    if (PetscLikely(n)) {
      const auto bytes = n * sizeof(M);

      PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to memset a NULL pointer with size %zu != 0", n);
      if (stream || use_async) {
        PetscCallCUPM(cupmMemsetAsync(ptr, value, bytes, stream));
      } else {
        PetscCallCUPM(cupmMemset(ptr, value, bytes));
      }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename M>
  static PetscErrorCode PetscCUPMMemset(M *ptr, int value, std::size_t n) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemsetAsync(ptr, value, n));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename D>
  static PetscErrorCode PetscCUPMMemset2DAsync(D *ptr, std::size_t pitch, int value, std::size_t width, std::size_t height, cupmStream_t stream = nullptr)
  {
    static_assert(!std::is_void<D>::value, "");
    const auto pitch_bytes = pitch * sizeof(D);
    const auto width_bytes = width * sizeof(D);
    const auto size        = width_bytes * height;

    PetscFunctionBegin;
    if (PetscUnlikely(!size)) PetscFunctionReturn(PETSC_SUCCESS);
    PetscAssert(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to memset a NULL pointer with size %zu != 0", size);
    if (stream) {
      PetscCallCUPM(cupmMemset2DAsync(ptr, pitch_bytes, value, width_bytes, height, stream));
    } else {
      PetscCallCUPM(cupmMemset2D(ptr, pitch_bytes, value, width_bytes, height));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // these we can transparently wrap, no need to namespace it to Petsc
  template <typename M>
  PETSC_NODISCARD static cupmError_t cupmFreeAsync(M &ptr, cupmStream_t stream = nullptr) noexcept
  {
    static_assert(std::is_pointer<util::decay_t<M>>::value, "");
    static_assert(!std::is_const<M>::value, "");

    if (ptr) {
      auto cerr = interface_type::cupmFreeAsync(std::forward<M>(ptr), stream);

      ptr = nullptr;
      if (PetscUnlikely(cerr != cupmSuccess)) return cerr;
    }
    return cupmSuccess;
  }

  PETSC_NODISCARD static cupmError_t cupmFreeAsync(std::nullptr_t ptr, cupmStream_t stream = nullptr) { return interface_type::cupmFreeAsync(ptr, stream); }

  template <typename M>
  PETSC_NODISCARD static cupmError_t cupmFree(M &ptr) noexcept
  {
    return cupmFreeAsync(ptr);
  }

  PETSC_NODISCARD static cupmError_t cupmFree(std::nullptr_t ptr) { return cupmFreeAsync(ptr); }

  template <typename M>
  PETSC_NODISCARD static cupmError_t cupmFreeHost(M &ptr) noexcept
  {
    static_assert(std::is_pointer<util::decay_t<M>>::value, "");
    const auto cerr = interface_type::cupmFreeHost(std::forward<M>(ptr));
    ptr             = nullptr;
    return cerr;
  }

  PETSC_NODISCARD static cupmError_t cupmFreeHost(std::nullptr_t ptr) { return interface_type::cupmFreeHost(ptr); }

  // specific wrapper for device launch function, as the real function is a C routine and
  // doesn't have variable arguments. The actual mechanics of this are a bit complicated but
  // boils down to the fact that ultimately we pass a
  //
  // void *args[] = {(void*)&kernel_args...};
  //
  // to the kernel launcher. Since we pass void* this means implicit conversion does **not**
  // happen to the kernel arguments so we must do it ourselves here. This function does this in
  // 3 stages:
  // 1. Enumerate the kernel arguments (cupmLaunchKernel)
  // 2. Deduce the signature of func() and static_cast the kernel arguments to the type
  //    expected by func() using the enumeration above (deduceKernelCall)
  // 3. Form the void* array with the converted arguments and call cuda/hipLaunchKernel with
  //    it. (interface_type::cupmLaunchKernel)
  template <typename F, typename... Args>
  PETSC_NODISCARD static cupmError_t cupmLaunchKernel(F &&func, cupmDim3 gridDim, cupmDim3 blockDim, std::size_t sharedMem, cupmStream_t stream, Args &&...kernelArgs) noexcept
  {
    return deduceKernelCall(util::index_sequence_for<Args...>{}, std::forward<F>(func), std::move(gridDim), std::move(blockDim), std::move(sharedMem), std::move(stream), std::forward<Args>(kernelArgs)...);
  }

  template <std::size_t block_size = 256, std::size_t warp_size = 32, typename F, typename... Args>
  static PetscErrorCode PetscCUPMLaunchKernel1D(std::size_t n, std::size_t sharedMem, cupmStream_t stream, F &&func, Args &&...kernelArgs) noexcept
  {
    static_assert(block_size > 0, "");
    static_assert(warp_size > 0, "");
    // want block_size to be a multiple of the warp_size
    static_assert(block_size % warp_size == 0, "");
    const auto nthread = std::min(n, block_size);
    const auto nblock  = (n + block_size - 1) / block_size;

    PetscFunctionBegin;
    // if n = 0 then nthread = 0, which is not allowed. rather than letting the user try to
    // decipher cryptic 'cuda/hipErrorLaunchFailure' we explicitly check for zero here
    PetscAssert(nthread, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to launch kernel with grid/block size 0");
    PetscCallCUPM(cupmLaunchKernel(std::forward<F>(func), nblock, nthread, sharedMem, stream, std::forward<Args>(kernelArgs)...));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  template <typename S, typename D, typename = void>
  struct is_static_castable : std::false_type { };

  template <typename S, typename D>
  struct is_static_castable<S, D, util::void_t<decltype(static_cast<D>(std::declval<S>()))>> : std::true_type { };

  template <typename D, typename S>
  static constexpr util::enable_if_t<is_static_castable<S, D>::value, D> cast_to(S &&src) noexcept
  {
    return static_cast<D>(std::forward<S>(src));
  }

  template <typename D, typename S>
  static constexpr util::enable_if_t<!is_static_castable<S, D>::value, D> cast_to(S &&src) noexcept
  {
    return const_cast<D>(std::forward<S>(src));
  }

  template <typename F, typename... Args, std::size_t... Idx>
  PETSC_NODISCARD static cupmError_t deduceKernelCall(util::index_sequence<Idx...>, F &&func, cupmDim3 gridDim, cupmDim3 blockDim, std::size_t sharedMem, cupmStream_t stream, Args &&...kernelArgs) noexcept
  {
    // clang-format off
    return interface_type::template cupmLaunchKernel(
      std::forward<F>(func),
      std::move(gridDim), std::move(blockDim), std::move(sharedMem), std::move(stream),
      // can't static_cast() here since the function argument type may be cv-qualified, in
      // which case we would need to const_cast(). But you can only const_cast() indirect types
      // (pointers, references). So we need a SFINAE monster that is a static_cast() if
      // possible, and a const_cast() if not. We could just use a C-style cast which *would*
      // work here since it tries the following and uses the first one that succeeds:
      //
      // 1. const_cast()
      // 2. static_cast()
      // 3. static_cast() then const_cast()
      // 4. reinterpret_cast()...
      //
      // the issue however is the final reinterpret_cast(). We absolutely cannot get there
      // because doing so would silently hide a ton of bugs, for example casting a PetscScalar
      // * to double * in complex builds, a PetscInt * to int * in 64idx builds, etc.
      cast_to<typename util::func_traits<F>::template arg<Idx>::type>(std::forward<Args>(kernelArgs))...
    );
    // clang-format on
  }

  static PetscErrorCode PetscLogCUPMMemcpyTransfer(cupmMemcpyKind_t kind, std::size_t size) noexcept
  {
    PetscFunctionBegin;
    // only the explicit HTOD or DTOH are handled, since we either don't log the other cases
    // (yet) or don't know the direction
    if (kind == cupmMemcpyDeviceToHost) PetscCall(PetscLogGpuToCpu(static_cast<PetscLogDouble>(size)));
    else if (kind == cupmMemcpyHostToDevice) PetscCall(PetscLogCpuToGpu(static_cast<PetscLogDouble>(size)));
    else (void)size;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

  #define PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T) \
    PETSC_CUPM_IMPL_CLASS_HEADER(T); \
    using cupmReal_t   = typename ::Petsc::device::cupm::impl::Interface<T>::cupmReal_t; \
    using cupmScalar_t = typename ::Petsc::device::cupm::impl::Interface<T>::cupmScalar_t; \
    using ::Petsc::device::cupm::impl::Interface<T>::cupmScalarCast; \
    using ::Petsc::device::cupm::impl::Interface<T>::cupmScalarPtrCast; \
    using ::Petsc::device::cupm::impl::Interface<T>::cupmRealPtrCast; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMGetMemType; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMemset; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMemsetAsync; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMalloc; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMallocAsync; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMallocHost; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMemcpy; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMemcpyAsync; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMemcpy2D; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMemcpy2DAsync; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMMemset2DAsync; \
    using ::Petsc::device::cupm::impl::Interface<T>::cupmFree; \
    using ::Petsc::device::cupm::impl::Interface<T>::cupmFreeAsync; \
    using ::Petsc::device::cupm::impl::Interface<T>::cupmFreeHost; \
    using ::Petsc::device::cupm::impl::Interface<T>::cupmLaunchKernel; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscCUPMLaunchKernel1D; \
    using ::Petsc::device::cupm::impl::Interface<T>::PetscDeviceCopyModeToCUPMMemcpyKind

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCCUPMINTERFACE_HPP */
