#pragma once

#include <petscsystypes.h>
#include <petscfvtypes.h>

/* MANSEC = DM */

#if PetscDefined(HAVE_LIBCEED)
  #include <ceed.h>

PETSC_EXTERN PetscErrorCode PetscFVGetCeedBasis(PetscFV, CeedBasis *);
PETSC_EXTERN PetscErrorCode PetscFVSetCeed(PetscFV, Ceed);
#endif
