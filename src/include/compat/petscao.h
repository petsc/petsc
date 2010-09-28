#ifndef _COMPAT_PETSC_AO_H
#define _COMPAT_PETSC_AO_H

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define AOInitializePackage(p) (0)
#endif

#endif /* _COMPAT_PETSC_AO_H */
