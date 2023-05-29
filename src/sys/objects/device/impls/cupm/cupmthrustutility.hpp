#ifndef PETSC_CUPM_THRUST_UTILITY_HPP
#define PETSC_CUPM_THRUST_UTILITY_HPP

#if defined(__cplusplus)
  #include <petsclog.h>         // PetscLogGpuTimeBegin()/End()
  #include <petscerror.h>       // SETERRQ()
  #include <petscdevice_cupm.h> // PETSC_USING_NVCC

  #include <thrust/system_error.h>     // thrust::system_error
  #include <thrust/execution_policy.h> // thrust::cuda/hip::par

namespace Petsc
{

namespace device
{

namespace cupm
{

  #if PetscDefined(USING_NVCC)
    #if !defined(THRUST_VERSION)
      #error "THRUST_VERSION not defined!"
    #endif
    #if !PetscDefined(USE_DEBUG) && (THRUST_VERSION >= 101600)
      #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(thrust::cuda::par_nosync.on(s), __VA_ARGS__)
    #else
      #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(thrust::cuda::par.on(s), __VA_ARGS__)
    #endif
  #elif PetscDefined(USING_HCC) // rocThrust has no par_nosync
    #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(thrust::hip::par.on(s), __VA_ARGS__)
  #else
    #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(__VA_ARGS__)
  #endif

namespace detail
{

struct PetscLogGpuTimer {
  PetscLogGpuTimer() noexcept { PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeBegin()); }
  ~PetscLogGpuTimer() noexcept { PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeEnd()); }
};

} // namespace detail

  #define THRUST_CALL(...) \
    [&] { \
      const auto timer = ::Petsc::device::cupm::detail::PetscLogGpuTimer{}; \
      return PETSC_THRUST_CALL_PAR_ON(__VA_ARGS__); \
    }()

  #define PetscCallThrust(...) \
    do { \
      try { \
        { \
          __VA_ARGS__; \
        } \
      } catch (const thrust::system_error &ex) { \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Thrust error: %s", ex.what()); \
      } \
    } while (0)

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CUPM_THRUST_UTILITY_HPP
