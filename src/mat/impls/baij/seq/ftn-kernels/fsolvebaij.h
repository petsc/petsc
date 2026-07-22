#pragma once

#include <petscsys.h>
#if PetscDefined(USE_FORTRAN_KERNEL_SOLVEBAIJ)
  #if PetscDefined(HAVE_FORTRAN_CAPS)
    #define fortransolvebaij4_       FORTRANSOLVEBAIJ4
    #define fortransolvebaij4unroll_ FORTRANSOLVEBAIJ4UNROLL
  #elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
    #define fortransolvebaij4_       fortransolvebaij4
    #define fortransolvebaij4unroll_ fortransolvebaij4unroll
  #endif

PETSC_EXTERN void fortransolvebaij4_(const PetscInt *, void *, const PetscInt *, const PetscInt *, const PetscInt *, const void *, const void *, const void *);
PETSC_EXTERN void fortransolvebaij4unroll_(const PetscInt *, void *, const PetscInt *, const PetscInt *, const PetscInt *, const void *, const void *);

#endif
