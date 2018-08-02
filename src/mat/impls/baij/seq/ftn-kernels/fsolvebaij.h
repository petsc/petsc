
#if !defined(__FSOLVEBAIJ_H)
#include <petscsys.h>
#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortransolvebaij4_         FORTRANSOLVEBAIJ4
#define fortransolvebaij4unroll_   FORTRANSOLVEBAIJ4UNROLL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortransolvebaij4_          fortransolvebaij4
#define fortransolvebaij4unroll_    fortransolvebaij4unroll
#endif

PETSC_EXTERN void fortransolvebaij4_(const PetscInt*,void*,const PetscInt*,const PetscInt*,const PetscInt*,const void*,const void*,const void*);
PETSC_EXTERN void fortransolvebaij4unroll_(const PetscInt*,void*,const PetscInt*,const PetscInt*,const PetscInt*,const void*,const void*);

#endif
#endif



