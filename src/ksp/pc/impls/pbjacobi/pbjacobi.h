#pragma once

#include <petsc/private/pcimpl.h>

/*
   Private context (data structure) for the PBJacobi preconditioner.
*/
typedef struct {
  const MatScalar *diag;
  PetscInt         bs, mbs; /* block size (bs), and number of blocks (mbs) */
  Mat              diagPB;  /* the matrix made of the diagonal blocks if some shell smatrix provided it; otherwise, NULL. Need to destroy it after use */
  void            *spptr;   /* opaque pointer to a device data structure */
} PC_PBJacobi;

#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_CUDA(PC, Mat);
#endif

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_Kokkos(PC, Mat);
#endif

PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_Host(PC, Mat);
PETSC_INTERN PetscErrorCode PCDestroy_PBJacobi(PC);
