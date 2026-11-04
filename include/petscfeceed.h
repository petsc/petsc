#pragma once

#include <petscsystypes.h>
#include <petscfetypes.h>

/* MANSEC = DM */

#if PetscDefined(HAVE_LIBCEED)
  #include <ceed.h>

PETSC_EXTERN PetscErrorCode PetscFEGetCeedBasis(PetscFE, CeedBasis *);
PETSC_EXTERN PetscErrorCode PetscFESetCeed(PetscFE, Ceed);
#endif
