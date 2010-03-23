
#if !defined(__FMULTADD_H)
#include "petscsys.h"
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultaddaij_ FORTRANMULTADDAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultaddaij_ fortranmultaddaij
#endif
EXTERN_C_BEGIN
EXTERN void fortranmultaddaij_(PetscInt*,void*,PetscInt*,PetscInt*,const MatScalar*,void*,void*);
EXTERN_C_END
#endif
#endif

