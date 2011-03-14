
#if !defined(__FNORM_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_NORM)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortrannormsqr_    FORTRANNORMSQR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortrannormsqr_    fortrannormsqr
#endif
EXTERN_C_BEGIN
extern void fortrannormsqr_(void*,PetscInt*,void*);
EXTERN_C_END
#endif
#endif

