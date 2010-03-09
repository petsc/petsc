
#ifndef DOT
#include "petscsys.h"

EXTERN_C_BEGIN








#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define fortranxtimesy_ FORTRANXTIMESY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranxtimesy_ fortranxtimesy
#endif
EXTERN void fortranxtimesy_(void*,void*,void*,PetscInt*);
#endif

EXTERN_C_END


#endif
