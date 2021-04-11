#if !defined(PETSCDMCEED_H)
#define PETSCDMCEED_H

#include <petscdm.h>

#if defined(PETSC_HAVE_LIBCEED)
#include <ceed.h>

PETSC_EXTERN PetscErrorCode DMGetCeed(DM, Ceed *);
#endif

#endif
