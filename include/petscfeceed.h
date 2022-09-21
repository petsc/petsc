#ifndef PETSCFECEED_H
#define PETSCFECEED_H

#include <petscfe.h>

#if defined(PETSC_HAVE_LIBCEED)
  #include <ceed.h>

PETSC_EXTERN PetscErrorCode PetscFEGetCeedBasis(PetscFE, CeedBasis *);
PETSC_EXTERN PetscErrorCode PetscFESetCeed(PetscFE, Ceed);
#endif

#endif
