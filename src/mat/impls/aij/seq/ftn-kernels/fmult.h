
#if !defined(__FMULT_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmultaij_                FORTRANMULTAIJ
#define fortranmulttransposeaddaij_    FORTRANMULTTRANSPOSEADDAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmultaij_                fortranmultaij
#define fortranmulttransposeaddaij_    fortranmulttransposeaddaij
#endif

PETSC_EXTERN void fortranmultaij_(PetscInt*,const PetscScalar*,const PetscInt*,const PetscInt*,const MatScalar*,PetscScalar*);
PETSC_EXTERN void fortranmulttransposeaddaij_(PetscInt*,const void*,PetscInt*,PetscInt*,void*,void*);

#endif
#endif

