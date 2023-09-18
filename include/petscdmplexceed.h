#pragma once

#if !defined(PETSC_HAVE_LIBCEED)
  #error "PETSc not configured for libCEED; reconfigrue --with-libceed or --download-libceed"
#endif

#include <petscdmplex.h>
#include <ceed.h>

PETSC_EXTERN PetscErrorCode DMPlexGetCeedRestriction(DM, DMLabel, PetscInt, PetscInt, PetscInt, CeedElemRestriction *);
