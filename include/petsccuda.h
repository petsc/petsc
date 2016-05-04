#if !defined(__PETSCCUDA_H)
#define __PETSCCUDA_H

#include <petscvec.h>

PETSC_EXTERN PetscErrorCode VecCUDAGetArrayReadWrite(Vec v, PetscScalar **a);
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayReadWrite(Vec v, PetscScalar **a);

PETSC_EXTERN PetscErrorCode VecCUDAGetArrayRead(Vec v, PetscScalar **a);
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayRead(Vec v, PetscScalar **a);

PETSC_EXTERN PetscErrorCode VecCUDAGetArrayWrite(Vec v, PetscScalar **a);
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayWrite(Vec v, PetscScalar **a);

PETSC_EXTERN PetscErrorCode VecCUDAPlaceArray(Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecCUDAReplaceArray(Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode VecCUDAResetArray(Vec);

#endif
