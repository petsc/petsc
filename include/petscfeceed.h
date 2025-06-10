#pragma once

#include <petscfe.h>

/* MANSEC = DM */

#if defined(PETSC_HAVE_LIBCEED)
  #include <ceed.h>

PETSC_EXTERN PetscErrorCode PetscFEGetCeedBasis(PetscFE, CeedBasis *);
PETSC_EXTERN PetscErrorCode PetscFESetCeed(PetscFE, Ceed);
#endif
