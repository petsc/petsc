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

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 4 && \
     PETSC_VERSION_SUBMINOR == 0 && \
     PETSC_VERSION_RELEASE  == 1)
#define PETSC_240 1
#endif

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 3 && \
     PETSC_VERSION_RELEASE  == 1)
#include "compat/petsc233.h"
#define PETSC_233 1
#endif

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
#include "compat/petsc232.h"
#define PETSC_232 1
#endif

#endif /* !PETSC_COMPAT_H */
