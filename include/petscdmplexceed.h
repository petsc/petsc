#pragma once

#include <petscdmplex.h>
#include <ceed.h>

/* MANSEC = DM */

#if !defined(PETSC_HAVE_LIBCEED)
  #error "PETSc not configured for libCEED; reconfigrue --with-libceed or --download-libceed"
#endif

PETSC_EXTERN PetscErrorCode DMPlexGetCeedRestriction(DM, DMLabel, PetscInt, PetscInt, PetscInt, CeedElemRestriction *);
PETSC_EXTERN PetscErrorCode DMPlexCreateCeedRestrictionFVM(DM, CeedElemRestriction *, CeedElemRestriction *);
PETSC_EXTERN PetscErrorCode DMPlexCeedComputeGeometryFVM(DM, CeedVector);
PETSC_EXTERN PetscErrorCode DMPlexTSComputeRHSFunctionFVMCEED(DM, PetscReal, Vec, Vec, void *);
