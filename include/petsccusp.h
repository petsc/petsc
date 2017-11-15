#if !defined(__PETSCCUSP_H)
#define __PETSCCUSP_H

#include <petscsys.h>
#include <petscmath.h>
#include <petscvec.h>
#include <cusp/array1d.h>

PETSC_EXTERN PetscErrorCode VecCUSPGetArrayReadWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayReadWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);

PETSC_EXTERN PetscErrorCode VecCUSPGetArrayRead(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayRead(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);

PETSC_EXTERN PetscErrorCode VecCUSPGetArrayWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayWrite(Vec v, cusp::array1d<PetscScalar,cusp::device_memory> **a);

#endif
