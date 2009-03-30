#if !defined(PETSC_COMPAT_H)
#define PETSC_COMPAT_H

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

#if   PETSC_VERSION_(2,3,2)
#include "compat/petsc232.h"
#elif PETSC_VERSION_(2,3,3)
#include "compat/petsc233.h"
#elif PETSC_VERSION_(3,0,0)
#include "compat/petsc300.h"
#endif

#endif /* !PETSC_COMPAT_H */
