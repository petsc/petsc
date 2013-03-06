
#if !defined(__FWAXPY_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranwaxpy_ FORTRANWAXPY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranwaxpy_ fortranwaxpy
#endif
PETSC_EXTERN void fortranwaxpy_(PetscInt*,const PetscScalar*,const PetscScalar*,const PetscScalar*,PetscScalar*);
#endif
#endif
