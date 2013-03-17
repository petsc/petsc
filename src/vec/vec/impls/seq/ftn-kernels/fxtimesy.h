
#if !defined(__FMAXPY_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranxtimesy_ FORTRANXTIMESY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranxtimesy_ fortranxtimesy
#endif
PETSC_EXTERN void fortranxtimesy_(const void*,const void*,void*,const PetscInt*);
#endif
#endif

