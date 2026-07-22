#pragma once

#include <petscsys.h>
#if PetscDefined(USE_FORTRAN_KERNEL_AYPX)
  #if PetscDefined(HAVE_FORTRAN_CAPS)
    #define fortranaypx_ FORTRANAYPX
  #elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
    #define fortranaypx_ fortranaypx
  #endif
PETSC_EXTERN void fortranaypx_(const PetscInt *, const PetscScalar *, const PetscScalar *, PetscScalar *);
#endif
