#if !defined(PETSCCUBLAS_H)
#define PETSCCUBLAS_H

#include <cublas_v2.h>

PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle);

#endif
