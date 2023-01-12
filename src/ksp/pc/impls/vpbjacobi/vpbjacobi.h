#ifndef VPBJACOBI_H
#define VPBJACOBI_H

#include <petsc/private/pcimpl.h>

/*
   Private context (data structure) for the VPBJacobi preconditioner.
*/
typedef struct {
  PetscInt   nblocks, min_bs, max_bs; // Stats recorded during setup for viewing
  MatScalar *diag;                    /* on host */
  void      *spptr;                   /* offload to devices */
} PC_VPBJacobi;

#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_CUDA(PC);
#endif

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_Kokkos(PC);
#endif

PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_Host(PC);
PETSC_INTERN PetscErrorCode PCDestroy_VPBJacobi(PC);

#endif
