#if !defined(PETSCDEVICE_H)
#define PETSCDEVICE_H

#include <petscsys.h>
#include <petscdevicetypes.h>
#include <petscpkg_version.h>

#if PetscDefined(HAVE_CUDA)
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
#define CHKERRCUDA(cerr) do {                                           \
    const cudaError_t _p_cuda_err__ = cerr;                             \
    if (PetscUnlikely(_p_cuda_err__ != cudaSuccess)) {                  \
      const char *name  = cudaGetErrorName(_p_cuda_err__);              \
      const char *descr = cudaGetErrorString(_p_cuda_err__);            \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s", \
               (PetscErrorCode)_p_cuda_err__,name,descr);               \
    }                                                                   \
  } while (0)
#else /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */
#define CHKERRCUDA(cerr) do {                                   \
    const cudaError_t _p_cuda_err__ = cerr;                     \
    if (PetscUnlikely(_p_cuda_err__ != cudaSuccess)) {          \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d",   \
               (PetscErrorCode)_p_cuda_err__);                  \
    }                                                           \
  } while (0)
#endif /* PETSC_PKG_CUDA_VERSION_GE(8,0,0) */

#define CHKERRCUBLAS(stat)   do {                                       \
    const cublasStatus_t _p_cublas_stat__ = stat;                       \
    if (PetscUnlikely(_p_cublas_stat__ != CUBLAS_STATUS_SUCCESS)) {     \
      const char *name = PetscCUBLASGetErrorName(_p_cublas_stat__);     \
      if (((_p_cublas_stat__ == CUBLAS_STATUS_NOT_INITIALIZED) ||       \
           (_p_cublas_stat__ == CUBLAS_STATUS_ALLOC_FAILED))   &&       \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                \
                 "cuBLAS error %d (%s). "                               \
                 "Reports not initialized or alloc failed; "            \
                 "this indicates the GPU may have run out resources",   \
                 (PetscErrorCode)_p_cublas_stat__,name);                \
      } else {                                                          \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuBLAS error %d (%s)",  \
                 (PetscErrorCode)_p_cublas_stat__,name);                \
      }                                                                 \
    }                                                                   \
  } while (0)

#define CHKERRCUSOLVER(stat) do {                                       \
    const cusolverStatus_t _p_cusolver_stat__ = stat;                   \
    if (PetscUnlikely(_p_cusolver_stat__ != CUSOLVER_STATUS_SUCCESS)) { \
      const char *name = PetscCUSolverGetErrorName(_p_cusolver_stat__); \
      if (((_p_cusolver_stat__ == CUSOLVER_STATUS_NOT_INITIALIZED) ||   \
           (_p_cusolver_stat__ == CUSOLVER_STATUS_ALLOC_FAILED)    ||   \
           (_p_cusolver_stat__ == CUSOLVER_STATUS_INTERNAL_ERROR)) &&   \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                \
                 "cuSolver error %d (%s). "                             \
                 "This indicates the GPU may have run out resources",   \
                 (PetscErrorCode)_p_cusolver_stat__,name);              \
      } else {                                                          \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,                         \
                 "cuSolver error %d (%s)",                              \
                 (PetscErrorCode)_p_cusolver_stat__,name);              \
      }                                                                 \
    }                                                                   \
  } while (0)

#define CHKERRCUFFT(res)     do {                                       \
    const cufftResult_t _p_cufft_stat__ = res;                          \
    if (PetscUnlikely(_p_cufft_stat__ != CUFFT_SUCCESS)) {              \
      const char *name = PetscCUFFTGetErrorName(_p_cufft_stat__);       \
      if (((_p_cufft_stat__ == CUFFT_SETUP_FAILED)  ||                  \
           (_p_cufft_stat__ == CUFFT_ALLOC_FAILED)) &&                  \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                \
                 "cuFFT error %d (%s). "                                \
                 "Reports not initialized or alloc failed; "            \
                 "this indicates the GPU has run out resources",        \
                 (PetscErrorCode)_p_cufft_stat__,name);                 \
      } else {                                                          \
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,                         \
                 "cuFFT error %d (%s)",                                 \
                 (PetscErrorCode)_p_cufft_stat__,name);                 \
      }                                                                 \
    }                                                                   \
  } while (0)

#define CHKERRCURAND(stat)   do {                                       \
    const curandStatus_t _p_curand_stat__ = stat;                       \
    if (PetscUnlikely(_p_curand_stat__ != CURAND_STATUS_SUCCESS)) {     \
      if (((_p_curand_stat__ == CURAND_STATUS_INITIALIZATION_FAILED) || \
           (_p_curand_stat__ == CURAND_STATUS_ALLOCATION_FAILED))    && \
          PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {                  \
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                \
                 "cuRAND error %d. "                                    \
                 "Reports not initialized or alloc failed; "            \
                 "this indicates the GPU has run out resources",        \
                 (PetscErrorCode)_p_curand_stat__);                     \
      } else {                                                          \
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,                         \
                 "cuRand error %d",(PetscErrorCode)_p_curand_stat__);   \
      }                                                                 \
    }                                                                   \
  } while (0)

PETSC_EXTERN cudaStream_t   PetscDefaultCudaStream; /* The default stream used by PETSc */

PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t*);
#endif /* PetscDefined(HAVE_CUDA) */

#if PetscDefined(HAVE_HIP)
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

#define CHKERRHIP(cerr)     do {                                        \
    const hipError_t _p_hip_err__ = cerr;                               \
    if (PetscUnlikely(_p_hip_err__ != hipSuccess)) {                    \
      const char *name  = hipGetErrorName(_p_hip_err__);                \
      const char *descr = hipGetErrorString(_p_hip_err__);              \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"hip error %d (%s) : %s",  \
               (PetscErrorCode)_p_hip_err__,name,descr);                \
    }                                                                   \
  } while (0)

#define CHKERRHIPBLAS(stat) do {                                        \
    const hipblasStatus_t _p_hipblas_stat__ = stat;                     \
    if (PetscUnlikely(_p_hipblas_stat__ != HIPBLAS_STATUS_SUCCESS)) {   \
      const char *name = PetscHIPBLASGetErrorName(_p_hipblas_stat__);   \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"hipBLAS error %d (%s)",   \
               (PetscErrorCode)_p_hipblas_stat__,name);                 \
    }                                                                   \
  } while (0)

/* TODO: SEK:  Need to figure out the hipsolver issues */
#define CHKERRHIPSOLVER(stat) do {                                      \
    const hipsolverStatus_t _p_hipsolver_stat__ = stat;                 \
    if (PetscUnlikely(_p_hipsolver_stat__ /* != HIPSOLVER_STATUS_SUCCESS */)) { \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"HIPSOLVER error %d",      \
               (PetscErrorCode)_p_hipsolver_stat__);                    \
    }                                                                   \
  } while (0)

/* hipSolver does not exist yet so we work around it
   rocSOLVER users rocBLAS for the handle
 * */
#if defined(__HIP_PLATFORM_NVCC__)
typedef cusolverDnHandle_t hipsolverHandle_t;
typedef cusolverStatus_t   hipsolverStatus_t;

/* Alias hipsolverDestroy to cusolverDnDestroy */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t *hipsolverhandle)
{
  return cusolverDnDestroy(hipsolverhandle)
}

/* Alias hipsolverCreate to cusolverDnCreate */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
  return cusolverDnCreate(hipsolverhandle)
}

/* Alias hipsolverGetStream to cusolverDnGetStream */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t *stream)
{
  return cusolverDnGetStream(handle,stream);
}

/* Alias hipsolverSetStream to cusolverDnSetStream */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t stream)
{
  return cusolveDnSetStream(handle,stream);
}
#else /* __HIP_PLATFORM_HCC__ */
typedef rocblas_handle hipsolverHandle_t;
typedef rocblas_status hipsolverStatus_t;

/* Alias hipsolverDestroy to rocblas_destroy_handle */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t  hipsolverhandle)
{
  return rocblas_destroy_handle(hipsolverhandle);
}

/* Alias hipsolverCreate to rocblas_destroy_handle */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
  return rocblas_create_handle(hipsolverhandle);
}

/* Alias hipsolverGetStream to rocblas_get_stream */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t *stream)
{
  return rocblas_get_stream(handle,stream);
}

/* Alias hipsolverSetStream to rocblas_set_stream */
PETSC_STATIC_INLINE hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t stream)
{
  return rocblas_set_stream(handle,stream);
}
#endif /* __HIP_PLATFORM_NVCC__ */
PETSC_EXTERN hipStream_t    PetscDefaultHipStream; /* The default stream used by PETSc */

PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t*);
#endif /* PetscDefined(HAVE_HIP) */

/* Cannot use the device context api without C++11 */
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
PETSC_EXTERN PetscErrorCode PetscDeviceInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDeviceFinalizePackage(void);

/* PetscDevice */
PETSC_EXTERN PetscErrorCode PetscDeviceInitialize(PetscDeviceType);
PETSC_EXTERN PetscBool      PetscDeviceInitialized(PetscDeviceType);
PETSC_EXTERN PetscErrorCode PetscDeviceCreate(PetscDeviceType,PetscInt,PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceConfigure(PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceView(PetscDevice,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDeviceDestroy(PetscDevice*);

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
#endif /* PetscDefined(HAVE_CXX_DIALECT_CXX11) */
#endif /* PETSCDEVICE_H */
