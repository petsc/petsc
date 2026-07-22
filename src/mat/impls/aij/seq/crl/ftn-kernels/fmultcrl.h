#pragma once

#include <petscsys.h>
#if PetscDefined(USE_FORTRAN_KERNEL_MULTCRL)
  #if PetscDefined(HAVE_FORTRAN_CAPS)
    #define fortranmultcrl_ FORTRANMULTCRL
  #elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
    #define fortranmultcrl_ fortranmultcrl
  #endif
PETSC_EXTERN void fortranmultcrl_(PetscInt *, PetscInt *, const PetscScalar *, PetscScalar *, PetscInt *, PetscScalar *);
#endif
