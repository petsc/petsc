
#if !defined(__FMAXPY_H)
#include "petscsys.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmaxpy4_ FORTRANMAXPY4
#define fortranmaxpy3_ FORTRANMAXPY3
#define fortranmaxpy2_ FORTRANMAXPY2
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmaxpy4_ fortranmaxpy4
#define fortranmaxpy3_ fortranmaxpy3
#define fortranmaxpy2_ fortranmaxpy2
#endif
EXTERN_C_BEGIN
EXTERN void fortranmaxpy4_(void*,void*,void*,void*,void*,const void*,const void*,const void*,const void*,PetscInt*);
EXTERN void fortranmaxpy3_(void*,void*,void*,void*,const void*,const void*,const void*,PetscInt*);
EXTERN void fortranmaxpy2_(void*,void*,void*,const void*,const void*,PetscInt*);
EXTERN_C_END
#endif
#endif

