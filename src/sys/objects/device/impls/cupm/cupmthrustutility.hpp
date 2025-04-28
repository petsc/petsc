#pragma once

#include <petsclog.h>         // PetscLogGpuTimeBegin()/End()
#include <petscsys.h>         // SETERRQ()
#include <petscdevice_cupm.h> // PETSC_USING_NVCC

#include <thrust/version.h>          // THRUST_VERSION
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
  #if THRUST_VERSION >= 101600
    #define PETSC_THRUST_HAS_ASYNC                 1
    #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(thrust::cuda::par_nosync.on(s), __VA_ARGS__)
  #else
    #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(thrust::cuda::par.on(s), __VA_ARGS__)
  #endif
#elif PetscDefined(USING_HCC)
  #if !defined(THRUST_VERSION)
    #error "THRUST_VERSION not defined!"
  #endif
  #if THRUST_VERSION >= 101600
    #define PETSC_THRUST_HAS_ASYNC                 1
    #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(thrust::hip::par_nosync.on(s), __VA_ARGS__)
  #else
    #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(thrust::hip::par.on(s), __VA_ARGS__)
  #endif
#else
  #define PETSC_THRUST_CALL_PAR_ON(func, s, ...) func(__VA_ARGS__)
#endif

#ifndef PETSC_THRUST_HAS_ASYNC
  #define PETSC_THRUST_HAS_ASYNC 0
#endif

namespace detail
{

struct PetscLogGpuTimer {
  PetscLogGpuTimer() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeBegin());
    PetscFunctionReturnVoid();
  }

  ~PetscLogGpuTimer() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeEnd());
    PetscFunctionReturnVoid();
  }
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
