#pragma once

#include <petscsys.h>
#if PetscDefined(USE_FORTRAN_KERNEL_SOLVEAIJ)
  #if PetscDefined(HAVE_FORTRAN_CAPS)
    #define fortransolveaij_ FORTRANSOLVEAIJ
  #elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
    #define fortransolveaij_ fortransolveaij
  #endif

PETSC_EXTERN void fortransolveaij_(const PetscInt *, void *, const PetscInt *, const PetscInt *, const PetscInt *, const void *, const void *);

#endif
