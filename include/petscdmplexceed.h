#if !defined(PETSCDMPLEXCEED_H)
#define PETSCDMPLEXCEED_H

#include <petscdmplex.h>

#if defined(PETSC_HAVE_LIBCEED)
#include <ceed.h>

PETSC_EXTERN PetscErrorCode DMPlexGetCeedRestriction(DM, CeedElemRestriction *);
#endif

#endif
