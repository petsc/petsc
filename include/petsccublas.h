#if !defined(PETSCCUBLAS_H)
#define PETSCCUBLAS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <petscsys.h>

#define WaitForGPU() PetscCUDASynchronize ? cudaDeviceSynchronize() : cudaSuccess;

/* CUDART_VERSION = 1000 x major + 10 x minor version */

/* Could not find exactly which CUDART_VERSION introduced cudaGetErrorName. At least it was in CUDA 8.0 (Sep. 2016) */
#if (CUDART_VERSION >= 8000) /* CUDA 8.0 */
#define CHKERRCUDA(cerr) \
do { \
   if (PetscUnlikely(cerr)) { \
      const char *name  = cudaGetErrorName(cerr); \
      const char *descr = cudaGetErrorString(cerr); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"cuda error %d (%s) : %s",(int)cerr,name,descr); \
   } \
} while(0)
#else
#define CHKERRCUDA(cerr) do {if (PetscUnlikely(cerr)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"cuda error %d",(int)cerr);} while(0)
#endif

#define CHKERRCUBLAS(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name = PetscCUBLASGetErrorName(stat); \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"cuBLAS error %d (%s)",(int)stat,name); \
   } \
} while(0)

PETSC_INTERN PetscErrorCode PetscCUBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscCUSOLVERDnInitializeHandle(void);

/* cuBLAS does not have cublasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscCUBLASGetErrorName(cublasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRCUBLAS macro */
PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t*);
#endif
