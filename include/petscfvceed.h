#ifndef PETSCFVCEED_H
#define PETSCFVCEED_H

#include <petscfv.h>

#if defined(PETSC_HAVE_LIBCEED)
  #include <ceed.h>

PETSC_EXTERN PetscErrorCode PetscFVGetCeedBasis(PetscFV, CeedBasis *);
PETSC_EXTERN PetscErrorCode PetscFVSetCeed(PetscFV, Ceed);
#endif

#endif
