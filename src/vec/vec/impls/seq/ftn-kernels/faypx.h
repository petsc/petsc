
#if !defined(__FAYPX_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranaypx_ FORTRANAYPX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranaypx_ fortranaypx
#endif
EXTERN_C_BEGIN
extern void fortranaypx_(PetscInt*,const PetscScalar*,const PetscScalar*,PetscScalar*); 
EXTERN_C_END
#endif
#endif
