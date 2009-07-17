
#if !defined(__FNORM_H)
#include "petsc.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortransolveaij_ FORTRANSOLVEAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortransolveaij_ fortransolveaij
#endif
EXTERN void fortransolveaij_(PetscInt*,PetscScalar*,const PetscInt*,const PetscInt*,const PetscInt*,const MatScalar*,const PetscScalar*);
#endif
#endif

