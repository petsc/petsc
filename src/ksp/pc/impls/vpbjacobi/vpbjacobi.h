#pragma once

#include <petsc/private/pcimpl.h>

/*
   Private context (data structure) for the VPBJacobi preconditioner.
*/
typedef struct {
  PetscInt   nblocks, min_bs, max_bs; // Stats recorded during setup for viewing
  MatScalar *diag;                    /* on host */
  Mat        diagVPB;                 /* the matrix made of the diagonal blocks if some shell matrix provided it; otherwise, NULL. Need to destroy it after use */
  void      *spptr;                   /* offload to devices */
} PC_VPBJacobi;

#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_CUDA(PC, Mat);
#endif

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_Kokkos(PC, Mat);
#endif

PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_Host(PC, Mat);
PETSC_INTERN PetscErrorCode PCDestroy_VPBJacobi(PC);
