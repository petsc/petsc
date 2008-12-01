#ifndef _PETSC_COMPAT_H
#define _PETSC_COMPAT_H

/* attribute recognised by some compilers to avoid 'unused' warnings */
#if !defined(PETSC_UNUSED)
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

#endif /* _PETSC_COMPAT_H */
