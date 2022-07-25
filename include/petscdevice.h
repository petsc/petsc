#if !defined(PETSCDEVICE_H)
#define PETSCDEVICE_H

#include <petscsys.h>
#include <petscdevicetypes.h>
#include <petscpkg_version.h>

#if defined(PETSC_HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cufft.h>

/* cuBLAS does not have cublasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscCUBLASGetErrorName(cublasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRCUBLAS macro */
PETSC_EXTERN const char* PetscCUSolverGetErrorName(cusolverStatus_t);
PETSC_EXTERN const char* PetscCUFFTGetErrorName(cufftResult);

/* REMOVE ME */
#define WaitForCUDA() cudaDeviceSynchronize()

/* CUDART_VERSION = 1000 x major + 10 x minor version */

/* Could not find exactly which CUDART_VERSION introduced cudaGetErrorName. At least it was in CUDA 8.0 (Sep. 2016) */
#if PETSC_PKG_CUDA_VERSION_GE(8,0,0)
#define PetscCallCUDA(...) do {                                         \
    const cudaError_t _p_cuda_err__ = __VA_ARGS__;                      \
    if (PetscUnlikely(_p_cuda_err__ != cudaSuccess)) {                  \
      const char *name  = cudaGetErrorName(_p_cuda_err__);              \
      const char *descr = cudaGetErrorString(_p_cuda_err__);            \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s",  \
              (PetscErrorCode)_p_cuda_err__,name,descr);                \
    }                                                                   \
  } while (0)
#else /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */
#define PetscCallCUDA(...) do {                                                                \
  const cudaError_t _p_cuda_err__ = __VA_ARGS__;                                               \
  PetscCheck(_p_cuda_err__ == cudaSuccess,PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d",(PetscErrorCode)_p_cuda_err__); \
} while (0)
#endif /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */
#define CHKERRCUDA(...) PetscCallCUDA(__VA_ARGS__)

#if PETSC_PKG_CUDA_VERSION_GE(8,0,0)
#define PetscCallCUDAVoid(...) do {                                     \
  const cudaError_t _p_cuda_err__ = __VA_ARGS__;                        \
  PetscCheckAbort(_p_cuda_err__ == cudaSuccess,PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s",(PetscErrorCode)_p_cuda_err__,cudaGetErrorName(_p_cuda_err__),cudaGetErrorString(_p_cuda_err__));  \
} while (0)
#else /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */
#define PetscCallCUDAVoid(...) do {                                     \
  const cudaError_t _p_cuda_err__ = __VA_ARGS__;                        \
  PetscCheckAbort(_p_cuda_err__ == cudaSuccess,PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d",(PetscErrorCode)_p_cuda_err__); \
} while (0)
#endif /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */

#define PetscCallCUBLAS(...) do {                                       \
    const cublasStatus_t _p_cublas_stat__ = __VA_ARGS__;                \
    if (PetscUnlikely(_p_cublas_stat__ != CUBLAS_STATUS_SUCCESS)) {     \
      const char *name = PetscCUBLASGetErrorName(_p_cublas_stat__);     \
      if (((_p_cublas_stat__ == CUBLAS_STATUS_NOT_INITIALIZED) ||       \
           (_p_cublas_stat__ == CUBLAS_STATUS_ALLOC_FAILED))   &&       \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                 \
                "cuBLAS error %d (%s). "                                \
                "Reports not initialized or alloc failed; "             \
                "this indicates the GPU may have run out resources",    \
                (PetscErrorCode)_p_cublas_stat__,name);                 \
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuBLAS error %d (%s)",   \
                (PetscErrorCode)_p_cublas_stat__,name);                 \
    }                                                                   \
  } while (0)
#define CHKERRCUBLAS(...) PetscCallCUBLAS(__VA_ARGS__)

#if (CUSPARSE_VER_MAJOR > 10 || CUSPARSE_VER_MAJOR == 10 && CUSPARSE_VER_MINOR >= 2) /* According to cuda/10.1.168 on OLCF Summit */
#define PetscCallCUSPARSE(...)\
do {\
  const cusparseStatus_t _p_cusparse_stat__ = __VA_ARGS__;\
  if (PetscUnlikely(_p_cusparse_stat__)) {\
    const char *name  = cusparseGetErrorName(_p_cusparse_stat__);\
    const char *descr = cusparseGetErrorString(_p_cusparse_stat__);\
    PetscCheck((_p_cusparse_stat__ != CUSPARSE_STATUS_NOT_INITIALIZED) && (_p_cusparse_stat__ != CUSPARSE_STATUS_ALLOC_FAILED),PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"cuSPARSE errorcode %d (%s) : %s. Reports not initialized or alloc failed; this indicates the GPU has run out resources",(int)_p_cusparse_stat__,name,descr); \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuSPARSE errorcode %d (%s) : %s",(int)_p_cusparse_stat__,name,descr);\
  }\
} while (0)
#else  /* (CUSPARSE_VER_MAJOR > 10 || CUSPARSE_VER_MAJOR == 10 && CUSPARSE_VER_MINOR >= 2) */
#define PetscCallCUSPARSE(...) do { \
  const cusparseStatus_t _p_cusparse_stat__ = __VA_ARGS__; \
  PetscCheck(_p_cusparse_stat__ == CUSPARSE_STATUS_SUCCESS,PETSC_COMM_SELF,PETSC_ERR_GPU,"cuSPARSE errorcode %d",(PetscErrorCode)_p_cusparse_stat__); \
  } while (0)
#endif /* (CUSPARSE_VER_MAJOR > 10 || CUSPARSE_VER_MAJOR == 10 && CUSPARSE_VER_MINOR >= 2) */
#define CHKERRCUSPARSE(...) PetscCallCUSPARSE(__VA_ARGS__)

#define PetscCallCUSOLVER(...) do {                                     \
    const cusolverStatus_t _p_cusolver_stat__ = __VA_ARGS__;            \
    if (PetscUnlikely(_p_cusolver_stat__ != CUSOLVER_STATUS_SUCCESS)) { \
      const char *name = PetscCUSolverGetErrorName(_p_cusolver_stat__); \
      if (((_p_cusolver_stat__ == CUSOLVER_STATUS_NOT_INITIALIZED) ||   \
           (_p_cusolver_stat__ == CUSOLVER_STATUS_ALLOC_FAILED)    ||   \
           (_p_cusolver_stat__ == CUSOLVER_STATUS_INTERNAL_ERROR)) &&   \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                 \
                "cuSolver error %d (%s). "                              \
                "This indicates the GPU may have run out resources",    \
                (PetscErrorCode)_p_cusolver_stat__,name);               \
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuSolver error %d (%s)", \
                (PetscErrorCode)_p_cusolver_stat__,name);               \
    }                                                                   \
  } while (0)
#define CHKERRCUSOLVER(...) PetscCallCUSOLVER(__VA_ARGS__)

#define PetscCallCUFFT(...)   do {                                      \
    const cufftResult_t _p_cufft_stat__ = __VA_ARGS__;                  \
    if (PetscUnlikely(_p_cufft_stat__ != CUFFT_SUCCESS)) {              \
      const char *name = PetscCUFFTGetErrorName(_p_cufft_stat__);       \
      if (((_p_cufft_stat__ == CUFFT_SETUP_FAILED)  ||                  \
           (_p_cufft_stat__ == CUFFT_ALLOC_FAILED)) &&                  \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                 \
                "cuFFT error %d (%s). "                                 \
                "Reports not initialized or alloc failed; "             \
                "this indicates the GPU has run out resources",         \
                (PetscErrorCode)_p_cufft_stat__,name);                  \
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuFFT error %d (%s)",    \
                (PetscErrorCode)_p_cufft_stat__,name);                  \
    }                                                                   \
  } while (0)
#define CHKERRCUFFT(...) PetscCallCUFFT(__VA_ARGS__)

#define PetscCallCURAND(...)  do {                                      \
    const curandStatus_t _p_curand_stat__ = __VA_ARGS__;                \
    if (PetscUnlikely(_p_curand_stat__ != CURAND_STATUS_SUCCESS)) {     \
      if (((_p_curand_stat__ == CURAND_STATUS_INITIALIZATION_FAILED) || \
           (_p_curand_stat__ == CURAND_STATUS_ALLOCATION_FAILED))    && \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                 \
                "cuRAND error %d. "                                     \
                "Reports not initialized or alloc failed; "             \
                "this indicates the GPU has run out resources",         \
                (PetscErrorCode)_p_curand_stat__);                      \
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,                          \
                "cuRand error %d",(PetscErrorCode)_p_curand_stat__);    \
    }                                                                   \
  } while (0)
#define CHKERRCURAND(...) PetscCallCURAND(__VA_ARGS__)

PETSC_EXTERN cudaStream_t   PetscDefaultCudaStream; /* The default stream used by PETSc */

PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t*);
#endif /* PETSC_HAVE_CUDA */

#if defined(PETSC_HAVE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas.h>
#if defined(__HIP_PLATFORM_NVCC__)
#include <cusolverDn.h>
#else /* __HIP_PLATFORM_HCC__ */
#include <rocsolver.h>
#endif /* __HIP_PLATFORM_NVCC__ */

/* REMOVE ME */
#define WaitForHIP() hipDeviceSynchronize()

/* hipBLAS does not have hipblasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscHIPBLASGetErrorName(hipblasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPBLAS macro */

#define PetscCallHIP(...)     do {                                      \
    const hipError_t _p_hip_err__ = __VA_ARGS__;                        \
    if (PetscUnlikely(_p_hip_err__ != hipSuccess)) {                    \
      const char *name  = hipGetErrorName(_p_hip_err__);                \
      const char *descr = hipGetErrorString(_p_hip_err__);              \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"hip error %d (%s) : %s",   \
              (PetscErrorCode)_p_hip_err__,name,descr);                 \
    }                                                                   \
  } while (0)
#define CHKERRHIP(...) PetscCallHIP(__VA_ARGS__)

#define PetscCallHIPBLAS(...) do {                                      \
    const hipblasStatus_t _p_hipblas_stat__ = __VA_ARGS__;              \
    if (PetscUnlikely(_p_hipblas_stat__ != HIPBLAS_STATUS_SUCCESS)) {   \
      const char *name = PetscHIPBLASGetErrorName(_p_hipblas_stat__);   \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"hipBLAS error %d (%s)",    \
              (PetscErrorCode)_p_hipblas_stat__,name);                  \
    }                                                                   \
  } while (0)
#define CHKERRHIPBLAS(...) PetscCallHIPBLAS(__VA_ARGS__)

/* TODO: SEK:  Need to figure out the hipsolver issues */
#define PetscCallHIPSOLVER(...) do { \
    const hipsolverStatus_t _p_hipsolver_stat__ = __VA_ARGS__; \
    PetscCheck(!_p_hipsolver_stat__,PETSC_COMM_SELF,PETSC_ERR_GPU,"HIPSOLVER error %d",(PetscErrorCode)_p_hipsolver_stat__); \
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
  return cusolverDnGetStream(handle,stream);
}

/* Alias hipsolverSetStream to cusolverDnSetStream */
static inline hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t stream)
{
  return cusolveDnSetStream(handle,stream);
}
#else /* __HIP_PLATFORM_HCC__ */
typedef rocblas_handle hipsolverHandle_t;
typedef rocblas_status hipsolverStatus_t;

/* Alias hipsolverDestroy to rocblas_destroy_handle */
static inline hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t  hipsolverhandle)
{
  return rocblas_destroy_handle(hipsolverhandle);
}

/* Alias hipsolverCreate to rocblas_destroy_handle */
static inline hipsolverStatus_t hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
  return rocblas_create_handle(hipsolverhandle);
}

/* Alias hipsolverGetStream to rocblas_get_stream */
static inline hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t *stream)
{
  return rocblas_get_stream(handle,stream);
}

/* Alias hipsolverSetStream to rocblas_set_stream */
static inline hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t stream)
{
  return rocblas_set_stream(handle,stream);
}
#endif /* __HIP_PLATFORM_NVCC__ */
PETSC_EXTERN hipStream_t    PetscDefaultHipStream; /* The default stream used by PETSc */

PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t*);
#endif /* PETSC_HAVE_HIP */

/* Cannot use the device context api without C++ */
#if defined(PETSC_HAVE_CXX)
PETSC_EXTERN PetscErrorCode PetscDeviceInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDeviceFinalizePackage(void);

/* PetscDevice */
PETSC_EXTERN PetscErrorCode PetscDeviceInitialize(PetscDeviceType);
PETSC_EXTERN PetscBool      PetscDeviceInitialized(PetscDeviceType);
PETSC_EXTERN PetscErrorCode PetscDeviceCreate(PetscDeviceType,PetscInt,PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceConfigure(PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceView(PetscDevice,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDeviceDestroy(PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceGetDeviceId(PetscDevice,PetscInt*);

/* PetscDeviceContext */
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext,PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext,PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext,PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext,PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextFork(PetscDeviceContext,PetscInt,PetscDeviceContext**);
PETSC_EXTERN PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext,PetscInt,PetscDeviceContextJoinMode,PetscDeviceContext**);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm,const char[],PetscDeviceContext);
#else
#define PetscDeviceInitialize(...)  0
#define PetscDeviceInitialized(...) PETSC_FALSE
#endif /* PETSC_HAVE_CXX */

#endif /* PETSCDEVICE_H */
