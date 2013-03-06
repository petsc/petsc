
#if !defined(__FNORM_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_NORM)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortrannormsqr_    FORTRANNORMSQR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortrannormsqr_    fortrannormsqr
#endif
PETSC_EXTERN void fortrannormsqr_(void*,PetscInt*,void*);
#endif
#endif

