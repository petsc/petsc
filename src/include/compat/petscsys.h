#ifndef _COMPAT_PETSC_SYS_H
#define _COMPAT_PETSC_SYS_H

#if (PETSC_VERSION_(2,3,2))
static PETSC_UNUSED
FILE *PETSC_STDERR = 0;
#endif


#if (PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define PETSCSPRNG SPRNG
#endif

#endif /* _COMPAT_PETSC_SYS_H */
