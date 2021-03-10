#if !defined(PETSCHIPBLAS_H)
#define PETSCHIPBLAS_H

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <petscsys.h>
#ifdef __HIP_PLATFORM_NVCC__
#include <cusolverDn.h>
#else
#include <rocsolver.h>
#endif

#define WaitForHIP() PetscHIPSynchronize ? hipDeviceSynchronize() : hipSuccess;

/* hipSolver does not exist yet so we work around it
   rocSOLVER users rocBLAS for the handle
 * */
#ifdef __HIP_PLATFORM_NVCC__
typedef cusolverDnHandle_t hipsolverHandle_t;
typedef cusolverStatus_t hipsolverStatus_t;

/* Alias hipsolverDestroy to cusolverDnDestroy*/
PETSC_STATIC_INLINE cusolverStatus_t hipsolverDestroy(hipsolverHandle_t *hipsolverhandle)
{
       return cusolverDnDestroy(hipsolverhandle)
}

/* Alias hipsolverCreate to cusolverDnCreate*/
PETSC_STATIC_INLINE cusolverStatus_t hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
       return cusolverDnCreate(hipsolverhandle)
}

#else
typedef rocblas_handle hipsolverHandle_t;
typedef rocblas_status hipsolverStatus_t;

/* Alias hipsolverDestroy to rocblas_destroy_handle*/
PETSC_STATIC_INLINE rocblas_status hipsolverDestroy(rocblas_handle hipsolverhandle)
{
       return rocblas_destroy_handle(hipsolverhandle);
}

/* Alias hipsolverCreate to rocblas_destroy_handle*/
PETSC_STATIC_INLINE rocblas_status hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
       return rocblas_create_handle(hipsolverhandle);
}

#endif

#define CHKERRHIP(cerr) \
do { \
   if (PetscUnlikely(cerr)) { \
      const char *name  = hipGetErrorName(cerr); \
      const char *descr = hipGetErrorString(cerr); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"hip error %d (%s) : %s",(int)cerr,name,descr); \
   } \
} while (0)

#define CHKERRHIPBLAS(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name = PetscHIPBLASGetErrorName(stat); \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipBLAS error %d (%s)",(int)stat,name); \
   } \
} while (0)

PETSC_EXTERN hipStream_t    PetscDefaultHipStream; /* The default stream used by PETSc */
PETSC_INTERN PetscErrorCode PetscHIPBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscHIPSOLVERInitializeHandle(void);

/* hipBLAS does not have hipblasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscHIPBLASGetErrorName(hipblasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPBLAS macro */
PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t*);
#endif
