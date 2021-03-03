#ifndef PETSC4PY_COMPAT_CUDA_H
#define PETSC4PY_COMPAT_CUDA_H

#if !defined(PETSC_HAVE_CUDA)

#define PetscCUDAError do { \
    PetscFunctionBegin; \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires CUDA",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode  VecCreateSeqCUDAWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *V) {PetscCUDAError;}
PetscErrorCode VecCreateMPICUDAWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *vv) {PetscCUDAError;}
PetscErrorCode VecCreateSeqCUDAWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar gpuarray[],Vec*V) {PetscCUDAError;}
PetscErrorCode VecCreateMPICUDAWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar gpuarray[],Vec*V) {PetscCUDAError;}

#undef PetscCUDAError

#endif

#endif/*PETSC4PY_COMPAT_CUDA_H*/
