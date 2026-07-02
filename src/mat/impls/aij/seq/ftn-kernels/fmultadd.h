#pragma once

#include <petscsys.h>
#if PetscDefined(USE_FORTRAN_KERNEL_MULTADDAIJ)
  #if PetscDefined(HAVE_FORTRAN_CAPS)
    #define fortranmultaddaij_ FORTRANMULTADDAIJ
  #elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
    #define fortranmultaddaij_ fortranmultaddaij
  #endif

PETSC_EXTERN void fortranmultaddaij_(PetscInt *, const void *, const PetscInt *, const PetscInt *, const MatScalar *, void *, void *);

#endif
