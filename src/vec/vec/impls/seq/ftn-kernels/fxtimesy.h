
#if !defined(__FMAXPY_H)
#include "petscsys.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranxtimesy_ FORTRANXTIMESY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranxtimesy_ fortranxtimesy
#endif
EXTERN_C_BEGIN
EXTERN void fortranxtimesy_(const void*,const void*,void*,const PetscInt *);
EXTERN_C_END
#endif
#endif

