#if !defined(PETSCDEVICE_H)
#define PETSCDEVICE_H

#include <petscsys.h>
#include <petscdevicetypes.h>

#if PetscDefined(HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cufft.h>

PETSC_EXTERN cudaEvent_t petsc_gputimer_begin;
PETSC_EXTERN cudaEvent_t petsc_gputimer_end;

/* cuBLAS does not have cublasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscCUBLASGetErrorName(cublasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRCUBLAS macro */
PETSC_EXTERN const char* PetscCUSolverGetErrorName(cusolverStatus_t);
PETSC_EXTERN const char* PetscCUFFTGetErrorName(cufftResult);

#define WaitForCUDA() PetscCUDASynchronize ? cudaDeviceSynchronize() : cudaSuccess;

/* CUDART_VERSION = 1000 x major + 10 x minor version */

/* Could not find exactly which CUDART_VERSION introduced cudaGetErrorName. At least it was in CUDA 8.0 (Sep. 2016) */
#if (CUDART_VERSION >= 8000) /* CUDA 8.0 */
#define CHKERRCUDA(cerr)                                                \
  do {                                                                  \
    if (PetscUnlikely(cerr)) {                                          \
      const char *name  = cudaGetErrorName(cerr);                       \
      const char *descr = cudaGetErrorString(cerr);                     \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s", \
               (int)cerr,name,descr);                                   \
    }                                                                   \
  } while (0)
#else
#define CHKERRCUDA(cerr) do {if (PetscUnlikely(cerr)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d",(int)cerr);} while (0)
#endif /* CUDART_VERSION >= 8000 */

#define CHKERRCUBLAS(stat)                                              \
  do {                                                                  \
    if (PetscUnlikely(stat)) {                                          \
      const char *name = PetscCUBLASGetErrorName(stat);                 \
      if (((stat == CUBLAS_STATUS_NOT_INITIALIZED) || (stat == CUBLAS_STATUS_ALLOC_FAILED)) && PetscCUDAInitialized) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"cuBLAS error %d (%s). Reports not initialized or alloc failed; this indicates the GPU has run out resources",(int)stat,name); \
      else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuBLAS error %d (%s)",(int)stat,name); \
    }                                                                   \
  } while (0)

#define CHKERRCUSOLVER(stat)                                            \
  do {                                                                  \
    if (PetscUnlikely(stat)) {                                          \
      const char *name = PetscCUSolverGetErrorName(stat);               \
      if ((stat == CUSOLVER_STATUS_NOT_INITIALIZED) || (stat == CUSOLVER_STATUS_ALLOC_FAILED) || (stat == CUSOLVER_STATUS_INTERNAL_ERROR)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"cuSolver error %d (%s). This indicates the GPU has run out resources",(int)stat,name); \
      else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuSolver error %d (%s)",(int)stat,name); \
    }                                                                   \
  } while (0)

#define CHKERRCUFFT(res)                                                \
  do {                                                                  \
    if (PetscUnlikely(res)) {                                           \
      const char *name = PetscCUFFTGetErrorName(res);                   \
      if (((res == CUFFT_SETUP_FAILED) || (res == CUFFT_ALLOC_FAILED)) && PetscCUDAInitialized) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"cuFFT error %d (%s). Reports not initialized or alloc failed; this indicates the GPU has run out resources",(int)res,name); \
      else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuFFT error %d (%s)",(int)res,name); \
    }                                                                   \
  } while (0)

PETSC_EXTERN cudaStream_t   PetscDefaultCudaStream; /* The default stream used by PETSc */
PETSC_INTERN PetscErrorCode PetscCUBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscCUSOLVERDnInitializeHandle(void);

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

#define WaitForHIP() PetscHIPSynchronize ? hipDeviceSynchronize() : hipSuccess;

PETSC_EXTERN hipEvent_t petsc_gputimer_begin;
PETSC_EXTERN hipEvent_t petsc_gputimer_end;

/* hipBLAS does not have hipblasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscHIPBLASGetErrorName(hipblasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPBLAS macro */

#define CHKERRHIP(cerr)                                                 \
  do {                                                                  \
    if (PetscUnlikely(cerr)) {                                          \
      const char *name  = hipGetErrorName(cerr);                        \
      const char *descr = hipGetErrorString(cerr);                      \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"hip error %d (%s) : %s",  \
               (int)cerr,name,descr);                                   \
    }                                                                   \
  } while (0)

#define CHKERRHIPBLAS(stat)                                             \
  do {                                                                  \
    if (PetscUnlikely(stat)) {                                          \
      const char *name = PetscHIPBLASGetErrorName(stat);                \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipBLAS error %d (%s)",   \
               (int)stat,name);                                         \
    }                                                                   \
  } while (0)

/* TODO: SEK:  Need to figure out the hipsolver issues */
#define CHKERRHIPSOLVER(err)                                            \
  do {                                                                  \
    if (PetscUnlikely(err)) {                                           \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSOLVER error %d",err); \
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
PETSC_INTERN PetscErrorCode PetscHIPBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscHIPSOLVERInitializeHandle(void);

PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t*);
#endif /* PetscDefined(HAVE_HIP) */

/* Cannot use the device context api without C++11 */
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
PETSC_EXTERN PetscErrorCode PetscDeviceInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDeviceFinalizePackage(void);

/* PetscDevice */
PETSC_EXTERN PetscErrorCode PetscDeviceCreate(PetscDeviceKind,PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceConfigure(PetscDevice);
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
