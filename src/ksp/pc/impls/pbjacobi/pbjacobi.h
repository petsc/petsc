#ifndef PBJACOBI_H
#define PBJACOBI_H

#include <petsc/private/pcimpl.h>

/*
   Private context (data structure) for the PBJacobi preconditioner.
*/
typedef struct {
  const MatScalar *diag;
  PetscInt         bs, mbs; /* block size (bs), and number of blocks (mbs) */
  void            *spptr;   /* opaque pointer to a device data structure */
} PC_PBJacobi;

#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_CUDA(PC);
#endif

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_Kokkos(PC);
#endif

PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_Host(PC);
PETSC_INTERN PetscErrorCode PCDestroy_PBJacobi(PC);

#endif
