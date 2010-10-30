#ifndef _COMPAT_PETSC_KSP_H
#define _COMPAT_PETSC_KSP_H

#include "private/kspimpl.h"

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0))
#define KSPNGMRES  "ngmres"
#define KSPSPECEST "specest"
#define KSPSetPCSide KSPSetPreconditionerSide
#define KSPGetPCSide KSPGetPreconditionerSide
#define KSP_NORM_NONE KSP_NORM_NO
#endif

#if (PETSC_VERSION_(3,0,0))
#define KSPBROYDEN "broyden"
#define KSPGCR     "gcr"
#endif

#endif /* _COMPAT_PETSC_KSP_H */
