#ifndef _COMPAT_PETSC_VEC_H
#define _COMPAT_PETSC_VEC_H

#include "private/vecimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define VecSqrtAbs VecSqrt
#endif

#endif /* _COMPAT_PETSC_VEC_H */
