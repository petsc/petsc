#if !defined __TAO_LAPACK_H
#define __TAO_LAPACK_H
#include "petsc.h"

#if defined(PETSC_BLASLAPACK_STDCALL) 

#  if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
#  define LAPACKgesv_ SGESV
#  else
#  define LAPACKgesv_ DGESV
#  endif /* defined PETSC_USES_FORTRAN_SINGLE || defined PETSC_USE_SINGLE */


#elif defined(PETSC_BLASLAPACK_UNDERSCORE)

#  if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
#  define LAPACKgesv_ sgesv_
#  else
#  define LAPACKgesv_ dgesv_
#  endif /* defined PETSC_USES_FORTRAN_SINGLE || defined PETSC_USE_SINGLE */


#elif defined(PETSC_BLASLAPACK_CAPS)

#  if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
#  define LAPACKgesv_ SGESV
#  else
#  define LAPACKgesv_ DGESV
#  endif /* defined PETSC_USES_FORTRAN_SINGLE || defined PETSC_USE_SINGLE */


#else

#  if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
#  define LAPACKgesv_ sgesv
#  else
#  define LAPACKgesv_ dgesv
#  endif /* defined PETSC_USES_FORTRAN_SINGLE || defined PETSC_USE_SINGLE */

#endif /* defined (PETSC_BLAS_LAPACK_STDCALL) */

PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN

extern void LAPACKgesv_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN_C_END
PETSC_EXTERN_CXX_END

#endif /* defined __TAO_LAPACK_H */

