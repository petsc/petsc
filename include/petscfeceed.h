#pragma once

#include <petscfe.h>

/* MANSEC = DM */

#if PetscDefined(HAVE_LIBCEED)
  #include <ceed.h>

PETSC_EXTERN PetscErrorCode PetscFEGetCeedBasis(PetscFE, CeedBasis *);
PETSC_EXTERN PetscErrorCode PetscFESetCeed(PetscFE, Ceed);
#endif
