#if !defined(PETSCCUBLAS_H)
#define PETSCCUBLAS_H

#include <cublas_v2.h>
#include <cusolverDn.h>

PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t*);

#endif
