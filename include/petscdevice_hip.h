#pragma once

#include <petscdevice.h>
#include <petscpkg_version.h>

/* MANSEC = Sys */
/* SUBMANSEC = Device */

#if defined(__HCC__) || (defined(__clang__) && defined(__HIP__))
  #define PETSC_USING_HCC 1
#endif

#if PetscDefined(HAVE_HIP)
  #include <hip/hip_runtime.h>

  // Use HIP complex types in hipBLAS to match cupmScalarPtrCast().
  #if PETSC_PKG_HIP_VERSION_GE(6, 0, 0)
    #define HIPBLAS_V2
  #else
    #define ROCM_MATHLIBS_API_USE_HIP_COMPLEX
  #endif
  #include <hipblas/hipblas.h>
  #include <hipsparse/hipsparse.h>

  #if PETSC_PKG_HIP_VERSION_LT(5, 4, 0)
    #define HIPSPARSE_ORDER_COL HIPSPARSE_ORDER_COLUMN
  #endif

  #include <hipsolver/hipsolver.h>
  #include <hip/hip_complex.h> // for hipComplex, hipDoubleComplex

  /*MC
    WaitForHIP - Block the calling host thread until all previously queued work on the current HIP device has completed

    Synopsis:
    #include <petscdevice_hip.h>
    hipError_t WaitForHIP(void)

    Not Collective; No Fortran Support

    Level: developer

    Note:
    Thin convenience wrapper around `hipDeviceSynchronize()`. Marked for removal in favour of
    explicit `PetscDeviceContext` synchronization.

.seealso: `PetscDeviceContext`, `PetscDeviceContextSynchronize()`, `WaitForCUDA()`
M*/
  // REMOVE ME
  #define WaitForHIP() hipDeviceSynchronize()

/* hipBLAS, hipSPARSE and hipSolver does not have hip*GetErrorName(). We create one on our own. */
PETSC_EXTERN const char *PetscHIPBLASGetErrorName(hipblasStatus_t);     /* PETSC_EXTERN since it is exposed by the CHKERRHIPBLAS macro */
PETSC_EXTERN const char *PetscHIPSPARSEGetErrorName(hipsparseStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPSPARSE macro */
PETSC_EXTERN const char *PetscHIPSolverGetErrorName(hipsolverStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPSOLVER macro */

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

  #define PetscHIPCheckLaunch \
    do { \
      /* Check synchronous errors, i.e. pre-launch */ \
      PetscCallHIP(hipGetLastError()); \
      /* Check asynchronous errors, i.e. kernel failed (ULF) */ \
      PetscCallHIP(hipDeviceSynchronize()); \
    } while (0)

  #define PetscCallHIPBLAS(...) \
    do { \
      const hipblasStatus_t _p_hipblas_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_hipblas_stat__ != HIPBLAS_STATUS_SUCCESS)) { \
        const char *name = PetscHIPBLASGetErrorName(_p_hipblas_stat__); \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "hipBLAS error %d (%s)", (PetscErrorCode)_p_hipblas_stat__, name); \
      } \
    } while (0)
  #define CHKERRHIPBLAS(...) PetscCallHIPBLAS(__VA_ARGS__)

  #define PetscCallHIPSPARSE(...) \
    do { \
      const hipsparseStatus_t _p_hipsparse_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_hipsparse_stat__ != HIPSPARSE_STATUS_SUCCESS)) { \
        const char *name = PetscHIPSPARSEGetErrorName(_p_hipsparse_stat__); \
        PetscCheck((_p_hipsparse_stat__ != HIPSPARSE_STATUS_NOT_INITIALIZED) && (_p_hipsparse_stat__ != HIPSPARSE_STATUS_ALLOC_FAILED), PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, "hipSPARSE errorcode %d (%s): Reports not initialized or alloc failed; this indicates the GPU has run out resources", (int)_p_hipsparse_stat__, name); \
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "hipSPARSE errorcode %d (%s)", (int)_p_hipsparse_stat__, name); \
      } \
    } while (0)
  #define CHKERRHIPSPARSE(...) PetscCallHIPSPARSE(__VA_ARGS__)

  #define PetscCallHIPSOLVER(...) \
    do { \
      const hipsolverStatus_t _p_hipsolver_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_hipsolver_stat__ != HIPSOLVER_STATUS_SUCCESS)) { \
        const char *name = PetscHIPSolverGetErrorName(_p_hipsolver_stat__); \
        if ((_p_hipsolver_stat__ == HIPSOLVER_STATUS_NOT_INITIALIZED || _p_hipsolver_stat__ == HIPSOLVER_STATUS_ALLOC_FAILED || _p_hipsolver_stat__ == HIPSOLVER_STATUS_INTERNAL_ERROR) && PetscDeviceInitialized(PETSC_DEVICE_HIP)) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "hipSolver error %d (%s). " \
                  "This indicates the GPU may have run out resources", \
                  (PetscErrorCode)_p_hipsolver_stat__, name); \
        } else { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "hipSolver error %d (%s)", (PetscErrorCode)_p_hipsolver_stat__, name); \
        } \
      } \
    } while (0)
  #define CHKERRHIPSOLVER(...) PetscCallHIPSOLVER(__VA_ARGS__)

// REMOVE ME
PETSC_EXTERN hipStream_t    PetscDefaultHipStream; // The default stream used by PETSc
PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t *);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t *);
PETSC_EXTERN PetscErrorCode PetscGetCurrentHIPStream(hipStream_t *);

#endif // PETSC_HAVE_HIP

// these can also be defined in petscdevice_cuda.h so we undef and define them *only* if the
// current compiler is HCC. In this case if petscdevice_cuda.h is included first, the macros
// would already be defined, but they would be empty since we cannot be using NVCC at the same
// time.
#if PetscDefined(USING_HCC)
  #undef PETSC_HOST_DECL
  #undef PETSC_DEVICE_DECL
  #undef PETSC_KERNEL_DECL
  #undef PETSC_SHAREDMEM_DECL
  #undef PETSC_FORCEINLINE
  #undef PETSC_CONSTMEM_DECL

  #define PETSC_HOST_DECL      __host__
  #define PETSC_DEVICE_DECL    __device__
  #define PETSC_KERNEL_DECL    __global__
  #define PETSC_SHAREDMEM_DECL __shared__
  #define PETSC_FORCEINLINE    __forceinline__
  #define PETSC_CONSTMEM_DECL  __constant__
#endif

#if !defined(PETSC_HOST_DECL) // use HOST_DECL as canary
  #define PETSC_HOST_DECL
  #define PETSC_DEVICE_DECL
  #define PETSC_KERNEL_DECL
  #define PETSC_SHAREDMEM_DECL
  #define PETSC_FORCEINLINE inline
  #define PETSC_CONSTMEM_DECL
#endif

#if !PetscDefined(DEVICE_DEFINED_DECLS_PRIVATE)
  #define PETSC_DEVICE_DEFINED_DECLS_PRIVATE
  #define PETSC_HOSTDEVICE_DECL        PETSC_HOST_DECL PETSC_DEVICE_DECL
  #define PETSC_DEVICE_INLINE_DECL     PETSC_DEVICE_DECL PETSC_FORCEINLINE
  #define PETSC_HOSTDEVICE_INLINE_DECL PETSC_HOSTDEVICE_DECL PETSC_FORCEINLINE
#endif
