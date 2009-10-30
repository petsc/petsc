#if !defined(PETSC4PY_COMPAT_H)
#define PETSC4PY_COMPAT_H

#ifndef PETSC_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define PETSC_UNUSED __attribute__ ((__unused__))
#   else
#     define PETSC_UNUSED
#   endif
# elif defined(__ICC)
#   define PETSC_UNUSED __attribute__ ((__unused__))
# else
#   define PETSC_UNUSED
# endif
#endif

#if !defined(PETSC_VERSION_)
#define PETSC_VERSION_(MAJOR,MINOR,SUBMINOR) \
  ((PETSC_VERSION_MAJOR == (MAJOR)) &&       \
   (PETSC_VERSION_MINOR == (MINOR)) &&       \
   (PETSC_VERSION_SUBMINOR == (SUBMINOR)) && \
   (PETSC_VERSION_RELEASE  == 1))
#endif

#include "compat/mpi.h"
#include "compat/petsc.h"
#include "compat/petscsys.h"
#include "compat/petscviewer.h"
#include "compat/petscis.h"
#include "compat/petscvec.h"
#include "compat/petscmat.h"
#include "compat/petscpc.h"
#include "compat/petscksp.h"
#include "compat/petscsnes.h"
#include "compat/petscts.h"
#include "compat/petscda.h"

#endif /* !PETSC4PY_COMPAT_H */
