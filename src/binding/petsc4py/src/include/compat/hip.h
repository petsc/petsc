#ifndef PETSC4PY_COMPAT_HIP_H
#define PETSC4PY_COMPAT_HIP_H

#if !defined(PETSC_HAVE_HIP)

#define PetscHIPError do { \
    PetscFunctionBegin; \
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires HIP",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode VecCreateSeqHIPWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *V) {PetscHIPError;}
PetscErrorCode VecCreateMPIHIPWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *vv) {PetscHIPError;}
PetscErrorCode VecCreateSeqHIPWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar gpuarray[],Vec*V) {PetscHIPError;}
PetscErrorCode VecCreateMPIHIPWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar gpuarray[],Vec*V) {PetscHIPError;}

#undef PetscHIPError

#endif

#endif/*PETSC4PY_COMPAT_HIP_H*/
