#if !defined(__PETSCCUSP_H)
#define __PETSCCUSP_H

#include <petscvec.h>
#include <cusp/array1d.h>

PETSC_EXTERN PetscErrorCode VecCUSPGetArrayReadWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayReadWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);

PETSC_EXTERN PetscErrorCode VecCUSPGetArrayRead(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayRead(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);

PETSC_EXTERN PetscErrorCode VecCUSPGetArrayWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);

PETSC_EXTERN PetscErrorCode VecCUSPGetCUDAArrayReadWrite(Vec v, PetscScalar **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreCUDAArrayReadWrite(Vec v, PetscScalar **a);

PETSC_EXTERN PetscErrorCode VecCUSPGetCUDAArrayRead(Vec v, PetscScalar **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreCUDAArrayRead(Vec v, PetscScalar **a);

PETSC_EXTERN PetscErrorCode VecCUSPGetCUDAArrayWrite(Vec v, PetscScalar **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreCUDAArrayWrite(Vec v, PetscScalar **a);

PETSC_EXTERN PetscErrorCode VecCUSPPlaceArray(Vec, cusp::array1d<PetscScalar,cusp::device_memory>*);
PETSC_EXTERN PetscErrorCode VecCUSPReplaceArray(Vec, cusp::array1d<PetscScalar,cusp::device_memory>*);
PETSC_EXTERN PetscErrorCode VecCUSPResetArray(Vec);

#endif
