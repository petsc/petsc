#pragma once

#include <petscsys.h>
#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define fortrancopy_ FORTRANCOPY
  #define fortranzero_ FORTRANZERO
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define fortrancopy_ fortrancopy
  #define fortranzero_ fortranzero
#endif
PETSC_EXTERN void fortrancopy_(PetscInt *, PetscScalar *, PetscScalar *);
PETSC_EXTERN void fortranzero_(PetscInt *, PetscScalar *);
