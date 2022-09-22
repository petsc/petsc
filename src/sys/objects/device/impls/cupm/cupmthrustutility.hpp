#ifndef PETSC_CUPM_THRUST_UTILITY_HPP
#define PETSC_CUPM_THRUST_UTILITY_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#if defined(__cplusplus)
  #include <thrust/device_ptr.h>
  #include <thrust/transform.h>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

  #if PetscDefined(USING_NVCC)
    #if !defined(THRUST_VERSION)
      #error "THRUST_VERSION not defined!"
    #endif
    #if !PetscDefined(USE_DEBUG) && (THRUST_VERSION >= 101600)
      #define thrust_call_par_on(func, s, ...) func(thrust::cuda::par_nosync.on(s), __VA_ARGS__)
    #else
      #define thrust_call_par_on(func, s, ...) func(thrust::cuda::par.on(s), __VA_ARGS__)
    #endif
  #elif PetscDefined(USING_HCC) // rocThrust has no par_nosync
    #define thrust_call_par_on(func, s, ...) func(thrust::hip::par.on(s), __VA_ARGS__)
  #else
    #define thrust_call_par_on(func, s, ...) func(__VA_ARGS__)
  #endif

namespace detail
{

struct PetscLogGpuTimer {
  PetscLogGpuTimer() noexcept { PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeBegin()); }
  ~PetscLogGpuTimer() noexcept { PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeEnd()); }
};

struct private_tag { };

} // namespace detail

  #define THRUST_CALL(...) \
    [&] { \
      const auto timer = ::Petsc::device::cupm::impl::detail::PetscLogGpuTimer{}; \
      return thrust_call_par_on(__VA_ARGS__); \
    }()

  #define PetscCallThrust(...) \
    do { \
      try { \
        __VA_ARGS__; \
      } catch (const thrust::system_error &ex) { \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Thrust error: %s", ex.what()); \
      } \
    } while (0)

template <typename T, typename BinaryOperator>
struct shift_operator {
  const T *const       s;
  const BinaryOperator op;

  PETSC_HOSTDEVICE_DECL PETSC_FORCEINLINE auto operator()(T x) const PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(op(std::move(x), *s))
};

template <typename T, typename BinaryOperator>
static inline auto make_shift_operator(T *s, BinaryOperator &&op) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(shift_operator<T, BinaryOperator>{s, std::forward<BinaryOperator>(op)});

  #define PetscValidDevicePointer(ptr, argno) PetscAssert(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Null device pointer for " PetscStringize(ptr) " Argument #%d", argno);

// actual implementation that calls thrust, 2 argument version
template <DeviceType DT, typename FunctorType, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(detail::private_tag, typename Interface<DT>::cupmStream_t stream, FunctorType &&functor, PetscInt n, T *xinout, T *yin = nullptr))
{
  const auto xptr   = thrust::device_pointer_cast(xinout);
  const auto retptr = (yin && (yin != xinout)) ? thrust::device_pointer_cast(yin) : xptr;

  PetscFunctionBegin;
  PetscValidDevicePointer(xinout, 4);
  PetscCallThrust(THRUST_CALL(thrust::transform, stream, xptr, xptr + n, retptr, std::forward<FunctorType>(functor)));
  PetscFunctionReturn(0);
}

// actual implementation that calls thrust, 3 argument version
template <DeviceType DT, typename FunctorType, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(detail::private_tag, typename Interface<DT>::cupmStream_t stream, FunctorType &&functor, PetscInt n, T *xin, T *yin, T *zin))
{
  const auto xptr = thrust::device_pointer_cast(xin);

  PetscFunctionBegin;
  PetscValidDevicePointer(xin, 4);
  PetscValidDevicePointer(yin, 5);
  PetscValidDevicePointer(zin, 6);
  PetscCallThrust(THRUST_CALL(thrust::transform, stream, xptr, xptr + n, thrust::device_pointer_cast(yin), thrust::device_pointer_cast(zin), std::forward<FunctorType>(functor)));
  PetscFunctionReturn(0);
}

// one last intermediate function to check n, and log flops for everything
template <DeviceType DT, typename F, typename... T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(typename Interface<DT>::cupmStream_t stream, F &&functor, PetscInt n, T &&...rest))
{
  PetscFunctionBegin;
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "n %" PetscInt_FMT " must be >= 0", n);
  if (PetscLikely(n)) {
    PetscCall(ThrustApplyPointwise<DT>(detail::private_tag{}, stream, std::forward<F>(functor), n, std::forward<T>(rest)...));
    PetscCall(PetscLogGpuFlops(n));
  }
  PetscFunctionReturn(0);
}

// serves as setup to the real implementation above
template <DeviceType T, typename F, typename... Args>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(PetscDeviceContext dctx, F &&functor, PetscInt n, Args &&...rest))
{
  typename Interface<T>::cupmStream_t stream;

  PetscFunctionBegin;
  static_assert(sizeof...(Args) <= 3, "");
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &stream));
  PetscCall(ThrustApplyPointwise<T>(stream, std::forward<F>(functor), n, std::forward<Args>(rest)...));
  PetscFunctionReturn(0);
}

  #define PetscCallCUPM_(...) \
    do { \
      using interface               = Interface<DT>; \
      using cupmError_t             = typename interface::cupmError_t; \
      const auto cupmName           = []() { return interface::cupmName(); }; \
      const auto cupmGetErrorName   = [](cupmError_t e) { return interface::cupmGetErrorName(e); }; \
      const auto cupmGetErrorString = [](cupmError_t e) { return interface::cupmGetErrorString(e); }; \
      const auto cupmSuccess        = interface::cupmSuccess; \
      PetscCallCUPM(__VA_ARGS__); \
    } while (0)

template <DeviceType DT, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustSet(typename Interface<DT>::cupmStream_t stream, PetscInt n, T *ptr, const T *val))
{
  PetscFunctionBegin;
  PetscValidPointer(val, 4);
  if (n) {
    const auto size = n * sizeof(T);

    PetscValidDevicePointer(ptr, 3);
    if (*val == T{0}) {
      PetscCallCUPM_(Interface<DT>::cupmMemsetAsync(ptr, 0, size, stream));
    } else {
      const auto xptr = thrust::device_pointer_cast(ptr);

      PetscCallThrust(THRUST_CALL(thrust::fill, stream, xptr, xptr + n, *val));
      if (std::is_same<util::remove_cv_t<T>, PetscScalar>::value) {
        PetscCall(PetscLogCpuToGpuScalar(size));
      } else {
        PetscCall(PetscLogCpuToGpu(size));
      }
    }
  }
  PetscFunctionReturn(0);
}

  #undef PetscCallCUPM_
  #undef PetscValidDevicePointer

template <DeviceType DT, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustSet(PetscDeviceContext dctx, PetscInt n, T *ptr, const T *val))
{
  typename Interface<DT>::cupmStream_t stream;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, &stream));
  PetscCall(ThrustSet(stream, n, ptr, val));
  PetscFunctionReturn(0);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CUPM_THRUST_UTILITY_HPP
