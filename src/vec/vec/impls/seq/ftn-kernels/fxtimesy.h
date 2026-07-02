#pragma once

#include <petscsys.h>
#if PetscDefined(USE_FORTRAN_KERNEL_MAXPY)
  #if PetscDefined(HAVE_FORTRAN_CAPS)
    #define fortranxtimesy_ FORTRANXTIMESY
  #elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
    #define fortranxtimesy_ fortranxtimesy
  #endif
PETSC_EXTERN void fortranxtimesy_(const void *, const void *, void *, const PetscInt *);
#endif
