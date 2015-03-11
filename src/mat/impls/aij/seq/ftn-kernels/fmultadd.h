
#if !defined(__FMULTADD_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultaddaij_ FORTRANMULTADDAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultaddaij_ fortranmultaddaij
#endif

PETSC_EXTERN void fortranmultaddaij_(PetscInt*,const void*,const PetscInt*,const PetscInt*,const MatScalar*,void*,void*);

#endif
#endif

