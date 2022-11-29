#ifndef PETSCDEVICE_CUDA_H
#define PETSCDEVICE_CUDA_H

#include <petscdevice.h>
#include <petscpkg_version.h>

#if defined(__NVCC__) || defined(__CUDACC__)
  #define PETSC_USING_NVCC 1
#endif

#if PetscDefined(HAVE_CUDA)
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <cusolverDn.h>
  #include <cusolverSp.h>
  #include <cufft.h>

/* cuBLAS does not have cublasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char *PetscCUBLASGetErrorName(cublasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRCUBLAS macro */
PETSC_EXTERN const char *PetscCUSolverGetErrorName(cusolverStatus_t);
PETSC_EXTERN const char *PetscCUFFTGetErrorName(cufftResult);

  /* REMOVE ME */
  #define WaitForCUDA() cudaDeviceSynchronize()

  /* CUDART_VERSION = 1000 x major + 10 x minor version */

  /* Could not find exactly which CUDART_VERSION introduced cudaGetErrorName. At least it was in CUDA 8.0 (Sep. 2016) */
  #if PETSC_PKG_CUDA_VERSION_GE(8, 0, 0)
    #define PetscCallCUDAVoid(...) \
      do { \
        const cudaError_t _p_cuda_err__ = __VA_ARGS__; \
        PetscCheckAbort(_p_cuda_err__ == cudaSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "cuda error %d (%s) : %s", (PetscErrorCode)_p_cuda_err__, cudaGetErrorName(_p_cuda_err__), cudaGetErrorString(_p_cuda_err__)); \
      } while (0)

    #define PetscCallCUDA(...) \
      do { \
        const cudaError_t _p_cuda_err__ = __VA_ARGS__; \
        PetscCheck(_p_cuda_err__ == cudaSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "cuda error %d (%s) : %s", (PetscErrorCode)_p_cuda_err__, cudaGetErrorName(_p_cuda_err__), cudaGetErrorString(_p_cuda_err__)); \
      } while (0)
  #else /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */
    #define PetscCallCUDA(...) \
      do { \
        const cudaError_t _p_cuda_err__ = __VA_ARGS__; \
        PetscCheck(_p_cuda_err__ == cudaSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "cuda error %d", (PetscErrorCode)_p_cuda_err__); \
      } while (0)

    #define PetscCallCUDAVoid(...) \
      do { \
        const cudaError_t _p_cuda_err__ = __VA_ARGS__; \
        PetscCheckAbort(_p_cuda_err__ == cudaSuccess, PETSC_COMM_SELF, PETSC_ERR_GPU, "cuda error %d", (PetscErrorCode)_p_cuda_err__); \
      } while (0)
  #endif /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */
  #define CHKERRCUDA(...) PetscCallCUDA(__VA_ARGS__)

  #define PetscCUDACheckLaunch \
    do { \
      /* Check synchronous errors, i.e. pre-launch */ \
      PetscCallCUDA(cudaGetLastError()); \
      /* Check asynchronous errors, i.e. kernel failed (ULF) */ \
      PetscCallCUDA(cudaDeviceSynchronize()); \
    } while (0)

  #define PetscCallCUBLAS(...) \
    do { \
      const cublasStatus_t _p_cublas_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_cublas_stat__ != CUBLAS_STATUS_SUCCESS)) { \
        const char *name = PetscCUBLASGetErrorName(_p_cublas_stat__); \
        if (((_p_cublas_stat__ == CUBLAS_STATUS_NOT_INITIALIZED) || (_p_cublas_stat__ == CUBLAS_STATUS_ALLOC_FAILED)) && PetscDeviceInitialized(PETSC_DEVICE_CUDA)) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "cuBLAS error %d (%s). " \
                  "Reports not initialized or alloc failed; " \
                  "this indicates the GPU may have run out resources", \
                  (PetscErrorCode)_p_cublas_stat__, name); \
        } else { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "cuBLAS error %d (%s)", (PetscErrorCode)_p_cublas_stat__, name); \
        } \
      } \
    } while (0)
  #define CHKERRCUBLAS(...) PetscCallCUBLAS(__VA_ARGS__)

  #if (CUSPARSE_VER_MAJOR > 10 || CUSPARSE_VER_MAJOR == 10 && CUSPARSE_VER_MINOR >= 2) /* According to cuda/10.1.168 on OLCF Summit */
    #define PetscCallCUSPARSE(...) \
      do { \
        const cusparseStatus_t _p_cusparse_stat__ = __VA_ARGS__; \
        if (PetscUnlikely(_p_cusparse_stat__)) { \
          const char *name  = cusparseGetErrorName(_p_cusparse_stat__); \
          const char *descr = cusparseGetErrorString(_p_cusparse_stat__); \
          PetscCheck((_p_cusparse_stat__ != CUSPARSE_STATUS_NOT_INITIALIZED) && (_p_cusparse_stat__ != CUSPARSE_STATUS_ALLOC_FAILED), PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                     "cuSPARSE errorcode %d (%s) : %s.; " \
                     "this indicates the GPU has run out resources", \
                     (int)_p_cusparse_stat__, name, descr); \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "cuSPARSE errorcode %d (%s) : %s", (int)_p_cusparse_stat__, name, descr); \
        } \
      } while (0)
  #else /* (CUSPARSE_VER_MAJOR > 10 || CUSPARSE_VER_MAJOR == 10 && CUSPARSE_VER_MINOR >= 2) */
    #define PetscCallCUSPARSE(...) \
      do { \
        const cusparseStatus_t _p_cusparse_stat__ = __VA_ARGS__; \
        PetscCheck(_p_cusparse_stat__ == CUSPARSE_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_GPU, "cuSPARSE errorcode %d", (PetscErrorCode)_p_cusparse_stat__); \
      } while (0)
  #endif /* (CUSPARSE_VER_MAJOR > 10 || CUSPARSE_VER_MAJOR == 10 && CUSPARSE_VER_MINOR >= 2) */
  #define CHKERRCUSPARSE(...) PetscCallCUSPARSE(__VA_ARGS__)

  #define PetscCallCUSOLVER(...) \
    do { \
      const cusolverStatus_t _p_cusolver_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_cusolver_stat__ != CUSOLVER_STATUS_SUCCESS)) { \
        const char *name = PetscCUSolverGetErrorName(_p_cusolver_stat__); \
        if (((_p_cusolver_stat__ == CUSOLVER_STATUS_NOT_INITIALIZED) || (_p_cusolver_stat__ == CUSOLVER_STATUS_ALLOC_FAILED) || (_p_cusolver_stat__ == CUSOLVER_STATUS_INTERNAL_ERROR)) && PetscDeviceInitialized(PETSC_DEVICE_CUDA)) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "cuSolver error %d (%s). " \
                  "This indicates the GPU may have run out resources", \
                  (PetscErrorCode)_p_cusolver_stat__, name); \
        } else { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "cuSolver error %d (%s)", (PetscErrorCode)_p_cusolver_stat__, name); \
        } \
      } \
    } while (0)
  #define CHKERRCUSOLVER(...) PetscCallCUSOLVER(__VA_ARGS__)

  #define PetscCallCUFFT(...) \
    do { \
      const cufftResult_t _p_cufft_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_cufft_stat__ != CUFFT_SUCCESS)) { \
        const char *name = PetscCUFFTGetErrorName(_p_cufft_stat__); \
        if (((_p_cufft_stat__ == CUFFT_SETUP_FAILED) || (_p_cufft_stat__ == CUFFT_ALLOC_FAILED)) && PetscDeviceInitialized(PETSC_DEVICE_CUDA)) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "cuFFT error %d (%s). " \
                  "Reports not initialized or alloc failed; " \
                  "this indicates the GPU has run out resources", \
                  (PetscErrorCode)_p_cufft_stat__, name); \
        } else { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "cuFFT error %d (%s)", (PetscErrorCode)_p_cufft_stat__, name); \
        } \
      } \
    } while (0)
  #define CHKERRCUFFT(...) PetscCallCUFFT(__VA_ARGS__)

  #define PetscCallCURAND(...) \
    do { \
      const curandStatus_t _p_curand_stat__ = __VA_ARGS__; \
      if (PetscUnlikely(_p_curand_stat__ != CURAND_STATUS_SUCCESS)) { \
        if (((_p_curand_stat__ == CURAND_STATUS_INITIALIZATION_FAILED) || (_p_curand_stat__ == CURAND_STATUS_ALLOCATION_FAILED)) && PetscDeviceInitialized(PETSC_DEVICE_CUDA)) { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, \
                  "cuRAND error %d. " \
                  "Reports not initialized or alloc failed; " \
                  "this indicates the GPU has run out resources", \
                  (PetscErrorCode)_p_curand_stat__); \
        } else { \
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_GPU, "cuRand error %d", (PetscErrorCode)_p_curand_stat__); \
        } \
      } \
    } while (0)
  #define CHKERRCURAND(...) PetscCallCURAND(__VA_ARGS__)

PETSC_EXTERN cudaStream_t   PetscDefaultCudaStream; // The default stream used by PETSc
PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *);
PETSC_EXTERN PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *);

#endif // PETSC_HAVE_CUDA

// these can also be defined in petscdevice_hip.h so we undef and define them *only* if the
// current compiler is NVCC. In this case if petscdevice_hip.h is included first, the macros
// would already be defined, but they would be empty since we cannot be using HCC at the same
// time.
#if PetscDefined(USING_NVCC)
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

#ifndef PETSC_HOST_DECL // use HOST_DECL as canary
  #define PETSC_HOST_DECL
  #define PETSC_DEVICE_DECL
  #define PETSC_KERNEL_DECL
  #define PETSC_SHAREDMEM_DECL
  #define PETSC_FORCEINLINE inline
  #define PETSC_CONSTMEM_DECL
#endif

#ifndef PETSC_DEVICE_DEFINED_DECLS_PRIVATE
  #define PETSC_DEVICE_DEFINED_DECLS_PRIVATE
  #define PETSC_HOSTDEVICE_DECL        PETSC_HOST_DECL PETSC_DEVICE_DECL
  #define PETSC_DEVICE_INLINE_DECL     PETSC_DEVICE_DECL PETSC_FORCEINLINE
  #define PETSC_HOSTDEVICE_INLINE_DECL PETSC_HOSTDEVICE_DECL PETSC_FORCEINLINE
#endif

#endif // PETSCDEVICE_CUDA_H
