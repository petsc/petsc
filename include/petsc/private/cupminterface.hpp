#ifndef PETSCCUPMINTERFACE_HPP
#define PETSCCUPMINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>

#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
#  define PETSC_HAVE_CUPM 1
#endif

#if PetscDefined(HAVE_CUPM)
#  define PETSC_HOST_DECL       __host__
#  define PETSC_DEVICE_DECL     __device__ __forceinline__
#  define PETSC_KERNEL_DECL     __global__
#  define PETSC_SHAREDMEM_DECL  __shared__
#else
#  define PETSC_HOST_DECL
#  define PETSC_DEVICE_DECL
#  define PETSC_KERNEL_DECL
#  define PETSC_SHAREDMEM_DECL
#endif

#define PETSC_HOSTDEVICE_DECL PETSC_HOST_DECL PETSC_DEVICE_DECL

#if defined(__cplusplus)
#include <array>

namespace Petsc
{

namespace Device
{

namespace CUPM
{

// enum describing available cupm devices, this is used as the template parameter to any
// class subclassing the Interface or using it as a member variable
enum class DeviceType : int {
  CUDA,
  HIP
};

static constexpr std::array<const char*const,5> DeviceTypes = {
  "cuda",
  "hip",
  "Petsc::Device::CUPM::CUPMDeviceType",
  "Petsc::Device::CUPM::CUPMDeviceType::",
  nullptr
};

namespace Impl
{

// A backend agnostic PetscCallCUPM() function, this will only work inside the member
// functions of a class inheriting from CUPM::Interface. Thanks to __VA_ARGS__ templated
// functions can also be wrapped inline:
//
// PetscCallCUPM(foo<int,char,bool>());
#define PetscCallCUPM(...) do {                                         \
    const cupmError_t cerr_p_ = __VA_ARGS__;                            \
    if (PetscUnlikely(cerr_p_ != cupmSuccess)) {                        \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",    \
              cupmName(),static_cast<PetscErrorCode>(cerr_p_),          \
              cupmGetErrorName(cerr_p_),cupmGetErrorString(cerr_p_));   \
    }                                                                   \
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
#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(our_prefix,our_suffix,their_prefix,their_suffix) \
  static const auto PetscConcat(our_prefix,our_suffix) = PetscConcat(their_prefix,their_suffix)

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
#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(our_suffix,their_suffix) \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,our_suffix,PETSC_CUPM_PREFIX_L,their_suffix)

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
#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE(suffix)         \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(suffix,suffix)

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
#define PETSC_CUPM_ALIAS_FUNCTION_EXACT(our_prefix,our_suffix,their_prefix,their_suffix) \
  PETSC_ALIAS_FUNCTION(static constexpr PetscConcat(our_prefix,our_suffix),PetscConcat(their_prefix,their_suffix))

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
#define PETSC_CUPM_ALIAS_FUNCTION_COMMON(our_suffix,their_suffix)       \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupm,our_suffix,PETSC_CUPM_PREFIX_L,their_suffix)

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
#define PETSC_CUPM_ALIAS_FUNCTION(suffix) PETSC_CUPM_ALIAS_FUNCTION_COMMON(suffix,suffix)

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
#define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(our_prefix,our_suffix,their_prefix,their_suffix,N) \
  PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(static constexpr PetscConcat(our_prefix,our_suffix),PetscConcat(their_prefix,their_suffix),N)

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
#define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(our_suffix,their_suffix,N) \
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(cupm,our_suffix,PETSC_CUPM_PREFIX_L,their_suffix,N)

// Base class that holds functions and variables that don't require CUDA or HIP to be present
// on the system
template <DeviceType T>
struct InterfaceBase
{
  static const DeviceType type = T;

  PETSC_CXX_COMPAT_DECL(constexpr const char* cupmName())
  {
    static_assert(util::integral_value(DeviceType::CUDA) == 0,"");
    static_assert(util::integral_value(DeviceType::HIP)  == 1,"");
    return std::get<util::integral_value(T)>(DeviceTypes);
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 PetscDeviceType cupmDeviceTypeToPetscDeviceType())
  {
    switch (T) {
    case DeviceType::CUDA: return PETSC_DEVICE_CUDA;
    case DeviceType::HIP:  return PETSC_DEVICE_HIP;
    }
    PetscUnreachable();
    return PETSC_DEVICE_INVALID;
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 PetscMemType cupmDeviceTypeToPetscMemType())
  {
    switch (T) {
    case DeviceType::CUDA: return PETSC_MEMTYPE_CUDA;
    case DeviceType::HIP:  return PETSC_MEMTYPE_HIP;
    }
    PetscUnreachable();
    return PETSC_MEMTYPE_HOST;
  }
};

// declare the base class static member variables
template <DeviceType T> const DeviceType InterfaceBase<T>::type;

#define PETSC_CUPM_BASE_CLASS_HEADER(DEVICE_TYPE)                               \
  using base_type = Petsc::Device::CUPM::Impl::InterfaceBase<DEVICE_TYPE>;      \
  using base_type::type;                                                        \
  using base_type::cupmName;                                                    \
  using base_type::cupmDeviceTypeToPetscDeviceType;                             \
  using base_type::cupmDeviceTypeToPetscMemType

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <DeviceType T> struct Interface;

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPM_PREFIX_L cuda
#define PETSC_CUPM_PREFIX_U CUDA
template <>
struct Interface<DeviceType::CUDA> : InterfaceBase<DeviceType::CUDA>
{
  PETSC_CUPM_BASE_CLASS_HEADER(DeviceType::CUDA);

  // typedefs
  using cupmError_t        = cudaError_t;
  using cupmEvent_t        = cudaEvent_t;
  using cupmStream_t       = cudaStream_t;
  using cupmDeviceProp_t   = cudaDeviceProp;
  using cupmMemcpyKind_t   = cudaMemcpyKind;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorDeviceAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
#if PETSC_PKG_CUDA_VERSION_GE(11,1,0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorStubLibrary);
#else
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary,ErrorInsufficientDriver);
#endif
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName);
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString);
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError);

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties);
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags);
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags);

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate);
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord);
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags);
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent);
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery);
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize);

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free);
  PETSC_CUPM_ALIAS_FUNCTION(Malloc);
#if PETSC_PKG_CUDA_VERSION_GE(11,2,0)
  PETSC_CUPM_ALIAS_FUNCTION(FreeAsync);
  PETSC_CUPM_ALIAS_FUNCTION(MallocAsync);
#else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync,Free,1);
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync,Malloc,1);
#endif
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync);
  PETSC_CUPM_ALIAS_FUNCTION(MallocHost);
  PETSC_CUPM_ALIAS_FUNCTION(FreeHost);
  PETSC_CUPM_ALIAS_FUNCTION(MemsetAsync);

  // specific wrapper for device launch function to mimic the HIP API since it has the superior
  // interface here, and it's not worth it to write another macro just for this specific
  // use-case
  template <typename FunctionT, typename... KernelArgsT>
  PETSC_CXX_COMPAT_DECL(cudaError_t cupmLaunchKernel(FunctionT func, dim3 gridDim, dim3 blockDim, std::size_t sharedMem, cudaStream_t stream, KernelArgsT&&... kernelArgs))
  {
    void* args[] = {&kernelArgs...};
    return cudaLaunchKernel(&func,gridDim,blockDim,args,sharedMem,stream);
  }
};
#undef PETSC_CUPM_PREFIX_L
#undef PETSC_CUPM_PREFIX_U
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPM_PREFIX_L hip
#define PETSC_CUPM_PREFIX_U HIP
template <>
struct Interface<DeviceType::HIP> : InterfaceBase<DeviceType::HIP>
{
  PETSC_CUPM_BASE_CLASS_HEADER(DeviceType::HIP);

  // typedefs
  using cupmError_t        = hipError_t;
  using cupmEvent_t        = hipEvent_t;
  using cupmStream_t       = hipStream_t;
  using cupmDeviceProp_t   = hipDeviceProp_t;
  using cupmMemcpyKind_t   = hipMemcpyKind;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  // see https://github.com/ROCm-Developer-Tools/HIP/blob/develop/bin/hipify-perl
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorDeviceAlreadyInUse,ErrorContextAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
  // as of HIP v4.2 cudaErrorStubLibrary has no HIP equivalent
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary,ErrorInsufficientDriver);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName);
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString);
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError);

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties);
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags);
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags);

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate);
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord);
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags);
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent);
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery);
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize);

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free);
  PETSC_CUPM_ALIAS_FUNCTION(Malloc);
  // HIP has no hipFreeAsync
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync,Free,1);
  // HIP has no hipMallocAsync
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync,Malloc,1);
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync);
  // hipMallocHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION_COMMON(MallocHost,HostMalloc);
  // hipFreeHost is deprecated
  PETSC_CUPM_ALIAS_FUNCTION_COMMON(FreeHost,HostFree);
  PETSC_CUPM_ALIAS_FUNCTION(MemsetAsync);

  // kernel launching
  PETSC_CUPM_ALIAS_FUNCTION(LaunchKernel);
};
#undef PETSC_CUPM_PREFIX_L
#undef PETSC_CUPM_PREFIX_U
#endif // PetscDefined(HAVE_HIP)

#undef PETSC_CUPM_BASE_CLASS_HEADER

// shorthand for bringing all of the typedefs from the base Interface class into your own,
// it's annoying that c++ doesn't have a way to do this automatically
#define PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(base_name,T)        \
  using base_name = Petsc::Device::CUPM::Impl::Interface<T>;            \
  /* introspection */                                                   \
  using base_name::type;                                                \
  using base_name::cupmName;                                            \
  using base_name::cupmDeviceTypeToPetscDeviceType;                     \
  using base_name::cupmDeviceTypeToPetscMemType;                        \
  /* types */                                                           \
  using typename base_name::cupmError_t;                                \
  using typename base_name::cupmEvent_t;                                \
  using typename base_name::cupmStream_t;                               \
  using typename base_name::cupmDeviceProp_t;                           \
  using typename base_name::cupmMemcpyKind_t;                           \
  /* variables */                                                       \
  using base_name::cupmSuccess;                                         \
  using base_name::cupmErrorNotReady;                                   \
  using base_name::cupmErrorDeviceAlreadyInUse;                         \
  using base_name::cupmErrorSetOnActiveProcess;                         \
  using base_name::cupmErrorStubLibrary;                                \
  using base_name::cupmErrorNoDevice;                                   \
  using base_name::cupmStreamNonBlocking;                               \
  using base_name::cupmDeviceMapHost;                                   \
  using base_name::cupmMemcpyHostToDevice;                              \
  using base_name::cupmMemcpyDeviceToHost;                              \
  using base_name::cupmMemcpyDeviceToDevice;                            \
  using base_name::cupmMemcpyHostToHost;                                \
  using base_name::cupmMemcpyDefault;                                   \
  /* functions */                                                       \
  using base_name::cupmGetErrorName;                                    \
  using base_name::cupmGetErrorString;                                  \
  using base_name::cupmGetLastError;                                    \
  using base_name::cupmGetDeviceCount;                                  \
  using base_name::cupmGetDeviceProperties;                             \
  using base_name::cupmGetDevice;                                       \
  using base_name::cupmSetDevice;                                       \
  using base_name::cupmGetDeviceFlags;                                  \
  using base_name::cupmSetDeviceFlags;                                  \
  using base_name::cupmEventCreate;                                     \
  using base_name::cupmEventDestroy;                                    \
  using base_name::cupmEventRecord;                                     \
  using base_name::cupmEventSynchronize;                                \
  using base_name::cupmEventElapsedTime;                                \
  using base_name::cupmStreamCreate;                                    \
  using base_name::cupmStreamCreateWithFlags;                           \
  using base_name::cupmStreamDestroy;                                   \
  using base_name::cupmStreamWaitEvent;                                 \
  using base_name::cupmStreamQuery;                                     \
  using base_name::cupmStreamSynchronize;                               \
  using base_name::cupmDeviceSynchronize;                               \
  using base_name::cupmFree;                                            \
  using base_name::cupmFreeAsync;                                       \
  using base_name::cupmMalloc;                                          \
  using base_name::cupmMallocAsync;                                     \
  using base_name::cupmMemcpy;                                          \
  using base_name::cupmMemcpyAsync;                                     \
  using base_name::cupmMallocHost;                                      \
  using base_name::cupmFreeHost;                                        \
  using base_name::cupmMemsetAsync;                                     \
  using base_name::cupmLaunchKernel

} // namespace Impl

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif /* __cplusplus */

#endif /* PETSCCUPMINTERFACE_HPP */
