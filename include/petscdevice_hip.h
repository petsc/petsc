#ifndef PETSCDEVICE_HIP_H
#define PETSCDEVICE_HIP_H

#include <petscdevice.h>
#include <petscpkg_version.h>

#if defined(__HCC__) || (defined(__clang__) && defined(__HIP__))
  #define PETSC_USING_HCC 1
#endif

#if PetscDefined(HAVE_HIP)
  #include <hip/hip_runtime.h>

  #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
    #include <hipblas/hipblas.h>
  #else
    #include <hipblas.h>
  #endif

  #if defined(__HIP_PLATFORM_NVCC__)
    #include <cusolverDn.h>
  #else // __HIP_PLATFORM_HCC__
    #if PETSC_PKG_HIP_VERSION_GE(5, 2, 0)
      #include <rocsolver/rocsolver.h>
    #else
      #include <rocsolver.h>
    #endif
  #endif                       // __HIP_PLATFORM_NVCC__
  #include <hip/hip_complex.h> // for hipComplex, hipDoubleComplex

  // REMOVE ME
  #define WaitForHIP() hipDeviceSynchronize()

/* hipBLAS does not have hipblasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char *PetscHIPBLASGetErrorName(hipblasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPBLAS macro */

  #define PetscCallHIP(...) \
    do { \
      const hipError_t _p_hip_err__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_hip_err__ != hipSuccess)) { \
        const char *name  = hipGetErrorName(_p_hip_err__); \
        const char *descr = hipGetErrorString(_p_hip_err__); \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "hip error %d (%s) : %s", (PetscErrorCode)_p_hip_err__, name, descr); \
      } \
    } while (0)
  #define CHKERRHIP(...) PetscCallHIP(__VA_ARGS__)

  #define PetscCallHIPBLAS(...) \
    do { \
      const hipblasStatus_t _p_hipblas_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_hipblas_stat__ != HIPBLAS_STATUS_SUCCESS)) { \
        const char *name = PetscHIPBLASGetErrorName(_p_hipblas_stat__); \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "hipBLAS error %d (%s)", (PetscErrorCode)_p_hipblas_stat__, name); \
      } \
    } while (0)
  #define CHKERRHIPBLAS(...) PetscCallHIPBLAS(__VA_ARGS__)

  /* TODO: SEK:  Need to figure out the hipsolver issues */
  #define PetscCallHIPSOLVER(...) \
    do { \
      const hipsolverStatus_t _p_hipsolver_stat__ = __VA_ARGS__; \
      PetscCheck(!_p_hipsolver_stat__, PETSC_COMM_SELF, PETSC_ERR_GPU, "HIPSOLVER error %d", (PetscErrorCode)_p_hipsolver_stat__); \
    } while (0)
  #define CHKERRHIPSOLVER(...) PetscCallHIPSOLVER(__VA_ARGS__)

  /* hipSolver does not exist yet so we work around it
 rocSOLVER users rocBLAS for the handle
 * */
  #if defined(__HIP_PLATFORM_NVCC__)
typedef cusolverDnHandle_t hipsolverHandle_t;
typedef cusolverStatus_t   hipsolverStatus_t;

/* Alias hipsolverDestroy to cusolverDnDestroy */
static inline hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t *hipsolverhandle)
{
  return cusolverDnDestroy(hipsolverhandle);
}

/* Alias hipsolverCreate to cusolverDnCreate */
static inline hipsolverStatus_t hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
  return cusolverDnCreate(hipsolverhandle);
}

/* Alias hipsolverGetStream to cusolverDnGetStream */
static inline hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t *stream)
{
  return cusolverDnGetStream(handle, stream);
}

/* Alias hipsolverSetStream to cusolverDnSetStream */
static inline hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t stream)
{
  return cusolveDnSetStream(handle, stream);
}
  #else  /* __HIP_PLATFORM_HCC__ */
typedef rocblas_handle hipsolverHandle_t;
typedef rocblas_status hipsolverStatus_t;

/* Alias hipsolverDestroy to rocblas_destroy_handle */
static inline hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t hipsolverhandle)
{
  return rocblas_destroy_handle(hipsolverhandle);
}

/* Alias hipsolverCreate to rocblas_destroy_handle */
static inline hipsolverStatus_t hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
  return rocblas_create_handle(hipsolverhandle);
}

// Alias hipsolverGetStream to rocblas_get_stream
static inline hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t *stream)
{
  return rocblas_get_stream(handle, stream);
}

// Alias hipsolverSetStream to rocblas_set_stream
static inline hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t stream)
{
  return rocblas_set_stream(handle, stream);
}
  #endif // __HIP_PLATFORM_NVCC__

// REMOVE ME
PETSC_EXTERN hipStream_t    PetscDefaultHipStream; // The default stream used by PETSc
PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t *);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t *);

#endif // PETSC_HAVE_HIP

// these can also be defined in petscdevice_cuda.h
#ifndef PETSC_DEVICE_DEFINED_DECLS_PRIVATE
  #define PETSC_DEVICE_DEFINED_DECLS_PRIVATE
  #if PetscDefined(USING_HCC)
    #define PETSC_HOST_DECL      __host__
    #define PETSC_DEVICE_DECL    __device__
    #define PETSC_KERNEL_DECL    __global__
    #define PETSC_SHAREDMEM_DECL __shared__
    #define PETSC_FORCEINLINE    __forceinline__
    #define PETSC_CONSTMEM_DECL  __constant__
  #else
    #define PETSC_HOST_DECL
    #define PETSC_DEVICE_DECL
    #define PETSC_KERNEL_DECL
    #define PETSC_SHAREDMEM_DECL
    #define PETSC_FORCEINLINE inline
    #define PETSC_CONSTMEM_DECL
  #endif // PETSC_USING_NVCC

  #define PETSC_HOSTDEVICE_DECL        PETSC_HOST_DECL PETSC_DEVICE_DECL
  #define PETSC_DEVICE_INLINE_DECL     PETSC_DEVICE_DECL PETSC_FORCEINLINE
  #define PETSC_HOSTDEVICE_INLINE_DECL PETSC_HOSTDEVICE_DECL PETSC_FORCEINLINE
#endif // PETSC_DEVICE_DEFINED_DECLS_PRIVATE

#endif // PETSCDEVICE_HIP_H
