#pragma once

#include <petscsys.h>
#if PetscDefined(USE_FORTRAN_KERNEL_WAXPY)
  #if PetscDefined(HAVE_FORTRAN_CAPS)
    #define fortranwaxpy_ FORTRANWAXPY
  #elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
    #define fortranwaxpy_ fortranwaxpy
  #endif
PETSC_EXTERN void fortranwaxpy_(const PetscInt *, const PetscScalar *, const PetscScalar *, const PetscScalar *, PetscScalar *);
#endif
