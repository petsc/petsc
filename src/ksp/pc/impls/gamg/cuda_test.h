#if !defined(CUDA_TEST_H)
#define CUDA_TEST_H

#include <petscdevice.h>

PETSC_INTERN PetscErrorCode RamdonSetC(PetscInt n, PetscReal *ramdom, PetscInt *permute, PetscBool *bIndexSet);
PETSC_INTERN PetscErrorCode getCOOValueC(PetscInt Istart, PetscInt Iend, PetscReal vfilter, const PetscInt *ia, const PetscInt *ja, const PetscScalar *aa, PetscInt *coo_i, PetscInt *coo_j, PetscScalar *coo_values);

#endif /* CUDA_TEST_H */


