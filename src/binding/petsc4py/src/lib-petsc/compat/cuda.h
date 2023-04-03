#ifndef PETSC4PY_COMPAT_CUDA_H
#define PETSC4PY_COMPAT_CUDA_H

#if !defined(PETSC_HAVE_CUDA)

#define PetscCUDAError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires CUDA",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode MatDenseCUDAGetArrayRead(Mat A, const PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDARestoreArrayRead(Mat A, const PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDAGetArrayWrite(Mat A, PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDARestoreArrayWrite(Mat A, PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDAGetArray(Mat A, PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatDenseCUDARestoreArray(Mat A, PetscScalar **gpuarray) {PetscCUDAError;}
PetscErrorCode MatCreateDenseCUDA(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar gpuarray[], Mat *A) {PetscCUDAError;}

#undef PetscCUDAError

#endif

#endif/*PETSC4PY_COMPAT_CUDA_H*/
