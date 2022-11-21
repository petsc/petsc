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

  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT() - declaration to alias a cuda/hip integral constant
  // value
  //
  // input params:
  // our_prefix   - the prefix of the alias
  // our_suffix   - the suffix of the alias
  // their_prefix - the prefix of the variable being aliased
  // their_suffix - the suffix of the variable being aliased
  //
  // example usage:
  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,Success,cuda,AllGood); ->
  // static const auto cupmSuccess = cudaAllGood;
  //
  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,Success,hip,AllRight); ->
  // static const auto cupmSuccess = hipAllRight;
  #define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(our_prefix, our_suffix, their_prefix, their_suffix) static const auto PetscConcat(our_prefix, our_suffix) = PetscConcat(their_prefix, their_suffix)

  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON() - declaration to alias a cuda/hip integral constant
  // value
  //
  // input params:
  // our_suffix   - the suffix of the alias
  // their_suffix - the suffix of the variable being aliased
  //
  // notes:
  // requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix
  //
  // example usage:
  // #define PETSC_CUPM_PREFIX_L cuda
  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(Success,AllGood); ->
  // static const auto cupmSuccess = cudaAllGood;
  //
  // #define PETSC_CUPM_PREFIX_L hip
  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(Success,AllRight); ->
  // static const auto cupmSuccess = hipAllRight;
  #define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(our_suffix, their_suffix) PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm, our_suffix, PETSC_CUPM_PREFIX_L, their_suffix)

  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE() - declaration to alias a cuda/hip integral constant value
  //
  // input param:
  // suffix - the common suffix shared between cuda, hip, and cupm
  //
  // notes:
  // requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix
  //
  // example usage:
  // #define PETSC_CUPM_PREFIX_L cuda
  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success); -> static const auto cupmSuccess = cudaSuccess;
  //
  // #define PETSC_CUPM_PREFIX_L hip
  // PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success); -> static const auto cupmSuccess = hipSuccess;
  #define PETSC_CUPM_ALIAS_INTEGRAL_VALUE(suffix) PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(suffix, suffix)

  // PETSC_CUPM_ALIAS_FUNCTION_EXACT() - declaration to alias a cuda/hip function
  //
  // input params:
  // our_prefix   - the prefix of the alias
  // our_suffix   - the suffix of the alias
  // their_prefix - the prefix of the function being aliased
  // their_suffix - the suffix of the function being aliased
  //
  // notes:
  // see PETSC_ALIAS_FUNCTION() for the exact nature of the expansion
  //
  // example usage:
  // PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupm,Malloc,cuda,Malloc) ->
  // template <typename... T>
  // static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
  // {
  //   return cudaMalloc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPM_ALIAS_FUNCTION_EXACT(our_prefix, our_suffix, their_prefix, their_suffix) PETSC_ALIAS_FUNCTION(static PetscConcat(our_prefix, our_suffix), PetscConcat(their_prefix, their_suffix))

  // PETSC_CUPM_ALIAS_FUNCTION_COMMON() - declaration to alias a cuda/hip function
  //
  // input params:
  // our_suffix   - the suffix of the alias
  // their_suffix - the common suffix of the cuda/hip function being aliased
  //
  // notes:
  // requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix of the function being
  // aliased. see PETSC_ALIAS_FUNCTION() for the exact nature of the expansion
  //
  // example usage:
  // #define PETSC_CUPM_PREFIX_L cuda
  // PETSC_CUPM_ALIAS_FUNCTION_COMMON(MallocFancy,Malloc) ->
  // template <typename... T>
  // static constexpr auto cupmMallocFancy(T&&... args) *noexcept and trailing return type deduction*
  // {
  //   return cudaMalloc(std::forward<T>(args)...);
  // }
  //
  // #define PETSC_CUPM_PREFIX_L hip
  // PETSC_CUPM_ALIAS_FUNCTION_COMMON(MallocFancy,Malloc) ->
  // template <typename... T>
  // static constexpr auto cupmMallocFancy(T&&... args) *noexcept and trailing return type deduction*
  // {
  //   return hipMalloc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPM_ALIAS_FUNCTION_COMMON(our_suffix, their_suffix) PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupm, our_suffix, PETSC_CUPM_PREFIX_L, their_suffix)

  // PETSC_CUPM_ALIAS_FUNCTION() - declaration to alias a cuda/hip function
  //
  // input param:
  // suffix - the common suffix for hip, cuda and the alias
  //
  // notes:
  // requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix of the function being
  // aliased. see PETSC_ALIAS_FUNCTION() for the exact nature of the expansion
  //
  // example usage:
  // #define PETSC_CUPM_PREFIX_L cuda
  // PETSC_CUPM_ALIAS_FUNCTION(Malloc) ->
  // template <typename... T>
  // static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
  // {
  //   return cudaMalloc(std::forward<T>(args)...);
  // }
  //
  // #define PETSC_CUPM_PREFIX_L hip
  // PETSC_CUPM_ALIAS_FUNCTION(Malloc) ->
  // template <typename... T>
  // static constexpr auto cupmMalloc(T&&... args) *noexcept and trailing return type deduction*
  // {
  //   return hipMalloc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPM_ALIAS_FUNCTION(suffix) PETSC_CUPM_ALIAS_FUNCTION_COMMON(suffix, suffix)

  // PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT() - declaration to alias a cuda/hip function but
  // discard the last N arguments
  //
  // input params:
  // our_prefix   - the prefix of the alias
  // our_suffix   - the suffix of the alias
  // their_prefix - the prefix of the function being aliased
  // their_suffix - the suffix of the function being aliased
  // N            - integer constant [0,INT_MAX) dictating how many arguments to chop off the end
  //
  // notes:
  // see PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS() for the exact nature of the expansion
  //
  // example use:
  // PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(cupm,MallocAsync,cuda,Malloc,1) ->
  // template <typename... T, typename Tend>
  // static constexpr auto cupmMallocAsync(T&&... args, Tend argend) *noexcept and trailing
  // return type deduction*
  // {
  //   (void)argend;
  //   return cudaMalloc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(our_prefix, our_suffix, their_prefix, their_suffix, N) PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(static PetscConcat(our_prefix, our_suffix), PetscConcat(their_prefix, their_suffix), N)

  // PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON() - declaration to alias a cuda/hip function but
  // discard the last N arguments
  //
  // input params:
  // our_suffix   - the suffix of the alias
  // their_suffix - the suffix of the function being aliased
  // N            - integer constant [0,INT_MAX) dictating how many arguments to chop off the end
  //
  // notes:
  // requires PETSC_CUPM_PREFIX_L to be defined to the specific prefix of the function being
  // aliased. see PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS() for the exact nature of the
  // expansion
  //
  // example use:
  // #define PETSC_CUPM_PREFIX_L cuda
  // PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync,Malloc,1) ->
  // template <typename... T, typename Tend>
  // static constexpr auto cupmMallocAsync(T&&... args, Tend argend) *noexcept and trailing
  // return type deduction*
  // {
  //   (void)argend;
  //   return cudaMalloc(std::forward<T>(args)...);
  // }
  #define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(our_suffix, their_suffix, N) PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(cupm, our_suffix, PETSC_CUPM_PREFIX_L, their_suffix, N)

// Base class that holds functions and variables that don't require CUDA or HIP to be present
// on the system
template <DeviceType T>
struct InterfaceBase {
  static const DeviceType type = T;

  PETSC_NODISCARD static constexpr const char *cupmName() noexcept
  {
    static_assert(util::integral_value(DeviceType::CUDA) == 0, "");
    static_assert(util::integral_value(DeviceType::HIP) == 1, "");
    return std::get<util::integral_value(T)>(DeviceTypes);
  }

  PETSC_NODISCARD static constexpr PetscDeviceType PETSC_DEVICE_CUPM() noexcept { return T == DeviceType::CUDA ? PETSC_DEVICE_CUDA : PETSC_DEVICE_HIP; }

  PETSC_NODISCARD static constexpr PetscMemType PETSC_MEMTYPE_CUPM() noexcept { return T == DeviceType::CUDA ? PETSC_MEMTYPE_CUDA : PETSC_MEMTYPE_HIP; }
};

// declare the base class static member variables
template <DeviceType T>
const DeviceType InterfaceBase<T>::type;

  #define PETSC_CUPM_BASE_CLASS_HEADER(base_name, DEVICE_TYPE) \
    using base_name = ::Petsc::device::cupm::impl::InterfaceBase<DEVICE_TYPE>; \
    using base_name::type; \
    using base_name::cupmName; \
    using base_name::PETSC_DEVICE_CUPM; \
    using base_name::PETSC_MEMTYPE_CUPM

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <DeviceType>
struct InterfaceImpl;

  #if PetscDefined(HAVE_CUDA)
    #define PETSC_CUPM_PREFIX_L cuda
    #define PETSC_CUPM_PREFIX_U CUDA
template <>
struct InterfaceImpl<DeviceType::CUDA> : InterfaceBase<DeviceType::CUDA> {
  PETSC_CUPM_BASE_CLASS_HEADER(base_type, DeviceType::CUDA);

  // typedefs
  using cupmError_t             = cudaError_t;
  using cupmEvent_t             = cudaEvent_t;
  using cupmStream_t            = cudaStream_t;
  using cupmDeviceProp_t        = cudaDeviceProp;
  using cupmMemcpyKind_t        = cudaMemcpyKind;
  using cupmComplex_t           = util::conditional_t<PetscDefined(USE_REAL_SINGLE), cuComplex, cuDoubleComplex>;
  using cupmPointerAttributes_t = struct cudaPointerAttributes;
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
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorDeviceAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
    #if PETSC_PKG_CUDA_VERSION_GE(11, 1, 0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorStubLibrary);
    #else
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary, ErrorInsufficientDriver);
    #endif
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeManaged);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(EventDisableTiming);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(HostAllocDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(HostAllocWriteCombined);
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemPoolAttrReleaseThreshold);
    #else
  static const cupmMemPoolAttr       cupmMemPoolAttrReleaseThreshold = 0;
    #endif

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName)
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString)
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError)

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties)
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(PointerGetAttributes)
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(DeviceGetMemPool)
  PETSC_CUPM_ALIAS_FUNCTION(MemPoolSetAttribute)
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
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate)
  PETSC_CUPM_ALIAS_FUNCTION(EventCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord)
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime)
  PETSC_CUPM_ALIAS_FUNCTION(EventQuery)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(StreamGetFlags)
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent)
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery)
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(GetSymbolAddress)

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free)
  PETSC_CUPM_ALIAS_FUNCTION(Malloc)
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(FreeAsync)
  PETSC_CUPM_ALIAS_FUNCTION(MallocAsync)
    #else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync, Free, 1)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync, Malloc, 1)
    #endif
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy)
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync)
  PETSC_CUPM_ALIAS_FUNCTION(MallocHost)
  PETSC_CUPM_ALIAS_FUNCTION(FreeHost)
  PETSC_CUPM_ALIAS_FUNCTION(Memset)
    #if PETSC_PKG_CUDA_VERSION_GE(11, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(MemsetAsync)
    #else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MemsetAsync, Memset, 1)
    #endif

  // launch control
  PETSC_CUPM_ALIAS_FUNCTION(LaunchHostFunc)
  template <typename FunctionT, typename... KernelArgsT>
  PETSC_NODISCARD static cudaError_t cupmLaunchKernel(FunctionT &&func, dim3 gridDim, dim3 blockDim, std::size_t sharedMem, cudaStream_t stream, KernelArgsT &&...kernelArgs) noexcept
  {
    void *args[] = {(void *)&kernelArgs...};
    return cudaLaunchKernel((void *)func, std::move(gridDim), std::move(blockDim), args, sharedMem, std::move(stream));
  }
};
    #undef PETSC_CUPM_PREFIX_L
    #undef PETSC_CUPM_PREFIX_U
  #endif // PetscDefined(HAVE_CUDA)

  #if PetscDefined(HAVE_HIP)
    #define PETSC_CUPM_PREFIX_L hip
    #define PETSC_CUPM_PREFIX_U HIP
template <>
struct InterfaceImpl<DeviceType::HIP> : InterfaceBase<DeviceType::HIP> {
  PETSC_CUPM_BASE_CLASS_HEADER(base_type, DeviceType::HIP);

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
  using cupmHostFn_t                                                 = void (*)(void *);
  using cupmMemPool_t                                                = void *;
  using cupmMemPoolAttr                                              = unsigned int;
    #endif

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  // see https://github.com/ROCm-Developer-Tools/HIP/blob/develop/bin/hipify-perl
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorDeviceAlreadyInUse, ErrorContextAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
  // as of HIP v4.2 cudaErrorStubLibrary has no HIP equivalent
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary, ErrorInsufficientDriver);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemoryTypeDevice);
  // see
  // https://github.com/ROCm-Developer-Tools/HIP/blob/develop/include/hip/hip_runtime_api.h#L156
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(MemoryTypeManaged, MemoryTypeUnified);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(EventDisableTiming);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(HostAllocDefault, HostMallocDefault);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(HostAllocWriteCombined, HostMallocWriteCombined);
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemPoolAttrReleaseThreshold);
    #else
  static const cupmMemPoolAttr       cupmMemPoolAttrReleaseThreshold = 0;
    #endif

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName)
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString)
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError)

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties)
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice)
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags)
  PETSC_CUPM_ALIAS_FUNCTION(PointerGetAttributes)
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(DeviceGetMemPool)
  PETSC_CUPM_ALIAS_FUNCTION(MemPoolSetAttribute)
    #else
  PETSC_NODISCARD static cupmError_t cupmDeviceGetMemPool(cupmMemPool_t *pool, int) noexcept
  {
    *pool = nullptr;
    return cupmSuccess;
  }

  PETSC_NODISCARD static cupmError_t cupmMemPoolSetAttribute(cupmMemPool_t, cupmMemPoolAttr, void *) noexcept { return cupmSuccess; }
    #endif
  PETSC_CUPM_ALIAS_FUNCTION(Init)

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate)
  PETSC_CUPM_ALIAS_FUNCTION(EventCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord)
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime)
  PETSC_CUPM_ALIAS_FUNCTION(EventQuery)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate)
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags)
  PETSC_CUPM_ALIAS_FUNCTION(StreamGetFlags)
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy)
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent)
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery)
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize)
  PETSC_CUPM_ALIAS_FUNCTION(GetSymbolAddress)

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free)
  PETSC_CUPM_ALIAS_FUNCTION(Malloc)
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(MallocAsync)
  PETSC_CUPM_ALIAS_FUNCTION(FreeAsync)
    #else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync, Malloc, 1)
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync, Free, 1)
    #endif
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy)
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync)
  // hipMallocHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION_COMMON(MallocHost, HostMalloc)
  // hipFreeHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION_COMMON(FreeHost, HostFree)
  PETSC_CUPM_ALIAS_FUNCTION(Memset)
  PETSC_CUPM_ALIAS_FUNCTION(MemsetAsync)

      // launch control
      // HIP appears to only have hipLaunchHostFunc from 5.2.0 onwards
      // https://github.com/ROCm-Developer-Tools/HIPIFY/blob/master/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md#7-execution-control=
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
  PETSC_CUPM_ALIAS_FUNCTION(LaunchHostFunc)
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
    #undef PETSC_CUPM_PREFIX_L
    #undef PETSC_CUPM_PREFIX_U
  #endif // PetscDefined(HAVE_HIP)

  // shorthand for bringing all of the typedefs from the base Interface class into your own,
  // it's annoying that c++ doesn't have a way to do this automatically
  #define PETSC_CUPM_IMPL_CLASS_HEADER(base_name, T) \
    PETSC_CUPM_BASE_CLASS_HEADER(PetscConcat(base_, base_name), T); \
    using base_name = ::Petsc::device::cupm::impl::InterfaceImpl<T>; \
    /* types */ \
    using typename base_name::cupmComplex_t; \
    using typename base_name::cupmError_t; \
    using typename base_name::cupmEvent_t; \
    using typename base_name::cupmStream_t; \
    using typename base_name::cupmDeviceProp_t; \
    using typename base_name::cupmMemcpyKind_t; \
    using typename base_name::cupmPointerAttributes_t; \
    using typename base_name::cupmMemoryType_t; \
    using typename base_name::cupmDim3; \
    using typename base_name::cupmMemPool_t; \
    using typename base_name::cupmMemPoolAttr; \
    /* variables */ \
    using base_name::cupmSuccess; \
    using base_name::cupmErrorNotReady; \
    using base_name::cupmErrorDeviceAlreadyInUse; \
    using base_name::cupmErrorSetOnActiveProcess; \
    using base_name::cupmErrorStubLibrary; \
    using base_name::cupmErrorNoDevice; \
    using base_name::cupmStreamDefault; \
    using base_name::cupmStreamNonBlocking; \
    using base_name::cupmDeviceMapHost; \
    using base_name::cupmMemcpyHostToDevice; \
    using base_name::cupmMemcpyDeviceToHost; \
    using base_name::cupmMemcpyDeviceToDevice; \
    using base_name::cupmMemcpyHostToHost; \
    using base_name::cupmMemcpyDefault; \
    using base_name::cupmMemoryTypeHost; \
    using base_name::cupmMemoryTypeDevice; \
    using base_name::cupmMemoryTypeManaged; \
    using base_name::cupmEventDisableTiming; \
    using base_name::cupmHostAllocDefault; \
    using base_name::cupmHostAllocWriteCombined; \
    using base_name::cupmMemPoolAttrReleaseThreshold; \
    /* functions */ \
    using base_name::cupmGetErrorName; \
    using base_name::cupmGetErrorString; \
    using base_name::cupmGetLastError; \
    using base_name::cupmGetDeviceCount; \
    using base_name::cupmGetDeviceProperties; \
    using base_name::cupmGetDevice; \
    using base_name::cupmSetDevice; \
    using base_name::cupmGetDeviceFlags; \
    using base_name::cupmSetDeviceFlags; \
    using base_name::cupmPointerGetAttributes; \
    using base_name::cupmDeviceGetMemPool; \
    using base_name::cupmMemPoolSetAttribute; \
    using base_name::cupmInit; \
    using base_name::cupmEventCreate; \
    using base_name::cupmEventCreateWithFlags; \
    using base_name::cupmEventDestroy; \
    using base_name::cupmEventRecord; \
    using base_name::cupmEventSynchronize; \
    using base_name::cupmEventElapsedTime; \
    using base_name::cupmEventQuery; \
    using base_name::cupmStreamCreate; \
    using base_name::cupmStreamCreateWithFlags; \
    using base_name::cupmStreamGetFlags; \
    using base_name::cupmStreamDestroy; \
    using base_name::cupmStreamWaitEvent; \
    using base_name::cupmStreamQuery; \
    using base_name::cupmStreamSynchronize; \
    using base_name::cupmDeviceSynchronize; \
    using base_name::cupmGetSymbolAddress; \
    using base_name::cupmMalloc; \
    using base_name::cupmMallocAsync; \
    using base_name::cupmMemcpy; \
    using base_name::cupmMemcpyAsync; \
    using base_name::cupmMallocHost; \
    using base_name::cupmMemset; \
    using base_name::cupmMemsetAsync; \
    using base_name::cupmLaunchHostFunc

template <DeviceType>
struct Interface;

// The actual interface class
template <DeviceType T>
struct Interface : InterfaceImpl<T> {
  PETSC_CUPM_IMPL_CLASS_HEADER(interface_type, T);

  using cupmReal_t   = util::conditional_t<PetscDefined(USE_REAL_SINGLE), float, double>;
  using cupmScalar_t = util::conditional_t<PetscDefined(USE_COMPLEX), cupmComplex_t, cupmReal_t>;

  // REVIEW ME: this needs to be cleaned up, it is unreadable
  PETSC_NODISCARD static constexpr cupmScalar_t makeCupmScalar(PetscScalar s) noexcept
  {
  #if PetscDefined(USE_COMPLEX)
    return cupmComplex_t{PetscRealPart(s), PetscImaginaryPart(s)};
  #else
    return static_cast<cupmReal_t>(s);
  #endif
  }

  PETSC_NODISCARD static constexpr const cupmScalar_t *cupmScalarCast(const PetscScalar *s) noexcept { return reinterpret_cast<const cupmScalar_t *>(s); }

  PETSC_NODISCARD static constexpr cupmScalar_t *cupmScalarCast(PetscScalar *s) noexcept { return reinterpret_cast<cupmScalar_t *>(s); }

  PETSC_NODISCARD static constexpr const cupmReal_t *cupmRealCast(const PetscReal *s) noexcept { return reinterpret_cast<const cupmReal_t *>(s); }

  PETSC_NODISCARD static constexpr cupmReal_t *cupmRealCast(PetscReal *s) noexcept { return reinterpret_cast<cupmReal_t *>(s); }

  #if !defined(PETSC_PKG_CUDA_VERSION_GE)
    #define PETSC_PKG_CUDA_VERSION_GE(...) 0
    #define CUPM_DEFINED_PETSC_PKG_CUDA_VERSION_GE
  #endif
  PETSC_NODISCARD static PetscErrorCode PetscCUPMGetMemType(const void *data, PetscMemType *type, PetscBool *registered = nullptr, PetscBool *managed = nullptr) noexcept
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
    PetscFunctionReturn(0);
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
  PETSC_NODISCARD static PetscErrorCode PetscCUPMMallocAsync(M **ptr, std::size_t n, cupmStream_t stream = nullptr) noexcept
  {
    static_assert(!std::is_void<M>::value, "");

    PetscFunctionBegin;
    PetscValidPointer(ptr, 1);
    if (PetscLikely(n)) {
      PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void **>(ptr), n * sizeof(M), stream));
    } else {
      *ptr = nullptr;
    }
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_NODISCARD static PetscErrorCode PetscCUPMMalloc(M **ptr, std::size_t n) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMallocAsync(ptr, n));
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_NODISCARD static PetscErrorCode PetscCUPMMallocHost(M **ptr, std::size_t n, unsigned int flags = cupmHostAllocDefault) noexcept
  {
    static_assert(!std::is_void<M>::value, "");

    PetscFunctionBegin;
    PetscValidPointer(ptr, 1);
    *ptr = nullptr;
    PetscCall(cupmMallocHost(reinterpret_cast<void **>(ptr), n * sizeof(M), flags));
    PetscFunctionReturn(0);
  }

  template <typename D, typename S = D>
  PETSC_NODISCARD static PetscErrorCode PetscCUPMMemcpyAsync(D *dest, const S *src, std::size_t n, cupmMemcpyKind_t kind, cupmStream_t stream = nullptr, bool use_async = false) noexcept
  {
    static_assert(sizeof(D) == sizeof(S), "");
    static_assert(!std::is_void<D>::value && !std::is_void<S>::value, "");
    const auto size = n * sizeof(D);

    PetscFunctionBegin;
    if (PetscUnlikely(!n)) PetscFunctionReturn(0);
    // cannot dereference (i.e. cannot call PetscValidPointer() here)
    PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
    PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
    // do early return after nullptr check since we need to check that they arent both nullptrs
    if (PetscUnlikely(dest == src)) PetscFunctionReturn(0);
    if (kind == cupmMemcpyHostToHost) {
      // If we are HTOH it is cheaper to check if the stream is idle and do a basic mempcy()
      // than it is to just call the vendor functions. This assumes of course that the stream
      // accounts for both memory regions being "idle"
      if (cupmStreamQuery(stream) == cupmSuccess) {
        PetscCall(PetscMemcpy(dest, src, size));
        PetscFunctionReturn(0);
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

    // only the explicit HTOD or DTOH are handled, since we either don't log the other cases
    // (yet) or don't know the direction
    if (kind == cupmMemcpyDeviceToHost) {
      PetscCall(PetscLogGpuToCpu(size));
    } else if (kind == cupmMemcpyHostToDevice) {
      PetscCall(PetscLogCpuToGpu(size));
    }
    PetscFunctionReturn(0);
  }

  template <typename D, typename S = D>
  PETSC_NODISCARD static PetscErrorCode PetscCUPMMemcpy(D *dest, const S *src, std::size_t n, cupmMemcpyKind_t kind) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemcpyAsync(dest, src, n, kind));
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_NODISCARD static PetscErrorCode PetscCUPMMemsetAsync(M *ptr, int value, std::size_t n, cupmStream_t stream = nullptr, bool use_async = false) noexcept
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
    PetscFunctionReturn(0);
  }

  template <typename M>
  PETSC_NODISCARD static PetscErrorCode PetscCUPMMemset(M *ptr, int value, std::size_t n) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscCUPMMemsetAsync(ptr, value, n));
    PetscFunctionReturn(0);
  }

  // these we can transparently wrap, no need to namespace it to Petsc
  template <typename M>
  PETSC_NODISCARD static cupmError_t cupmFreeAsync(M &&ptr, cupmStream_t stream = nullptr) noexcept
  {
    static_assert(std::is_pointer<util::decay_t<M>>::value, "");

    if (ptr) {
      auto cerr = interface_type::cupmFreeAsync(std::forward<M>(ptr), stream);

      ptr = nullptr;
      if (PetscUnlikely(cerr != cupmSuccess)) return cerr;
    }
    return cupmSuccess;
  }

  PETSC_NODISCARD static cupmError_t cupmFreeAsync(std::nullptr_t ptr, cupmStream_t stream = nullptr) noexcept { return interface_type::cupmFreeAsync(ptr, stream); }

  template <typename M>
  PETSC_NODISCARD static cupmError_t cupmFree(M &&ptr) noexcept
  {
    return cupmFreeAsync(std::forward<M>(ptr));
  }

  PETSC_NODISCARD static cupmError_t cupmFree(std::nullptr_t ptr) noexcept { return cupmFreeAsync(ptr); }

  template <typename M>
  PETSC_NODISCARD static cupmError_t cupmFreeHost(M &&ptr) noexcept
  {
    static_assert(std::is_pointer<util::decay_t<M>>::value, "");
    const auto cerr = interface_type::cupmFreeHost(std::forward<M>(ptr));
    ptr             = nullptr;
    return cerr;
  }

  PETSC_NODISCARD static cupmError_t cupmFreeHost(std::nullptr_t ptr) noexcept { return interface_type::cupmFreeHost(ptr); }

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
  PETSC_NODISCARD static PetscErrorCode PetscCUPMLaunchKernel1D(std::size_t n, std::size_t sharedMem, cupmStream_t stream, F &&func, Args &&...kernelArgs) noexcept
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
    PetscFunctionReturn(0);
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
      // which case we would need to const_cast(). But you can only const_cast()
      // indirect types (pointers, references) and I don't want to add a
      // static_cast_that_becomes_a_const_cast() SFINAE monster to this template mess. C-style
      // casts luckily work here since it tries the following and uses the first one that
      // succeeds:
      // 1. const_cast()
      // 2. static_cast()
      // 3. static_cast() then const_cast()
      // 4. reinterpret_cast()...
      // hopefully we never get to reinterpret_cast() land
      //(typename util::func_traits<F>::template arg<Idx>::type)(kernelArgs)...
      cast_to<typename util::func_traits<F>::template arg<Idx>::type>(std::forward<Args>(kernelArgs))...
    );
    // clang-format on
  }
};

  #define PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(base_name, T) \
    PETSC_CUPM_IMPL_CLASS_HEADER(PetscConcat(base_name, _impl), T); \
    using base_name = ::Petsc::device::cupm::impl::Interface<T>; \
    using typename base_name::cupmReal_t; \
    using typename base_name::cupmScalar_t; \
    using base_name::makeCupmScalar; \
    using base_name::cupmScalarCast; \
    using base_name::cupmRealCast; \
    using base_name::PetscCUPMGetMemType; \
    using base_name::PetscCUPMMemset; \
    using base_name::PetscCUPMMemsetAsync; \
    using base_name::PetscCUPMMalloc; \
    using base_name::PetscCUPMMallocAsync; \
    using base_name::PetscCUPMMallocHost; \
    using base_name::PetscCUPMMemcpy; \
    using base_name::PetscCUPMMemcpyAsync; \
    using base_name::cupmFree; \
    using base_name::cupmFreeAsync; \
    using base_name::cupmFreeHost; \
    using base_name::cupmLaunchKernel; \
    using base_name::PetscCUPMLaunchKernel1D; \
    using base_name::PetscDeviceCopyModeToCUPMMemcpyKind

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCCUPMINTERFACE_HPP */
