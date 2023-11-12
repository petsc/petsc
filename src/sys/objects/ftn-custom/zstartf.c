#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscinitializefortran_      PETSCINITIALIZEFORTRAN
  #define petscsetfortranbasepointers_ PETSCSETFORTRANBASEPOINTERS
  #define petsc_null_function_         PETSC_NULL_FUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscinitializefortran_      petscinitializefortran
  #define petscsetfortranbasepointers_ petscsetfortranbasepointers
  #define petsc_null_function_         petsc_null_function
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_null_function_ petsc_null_function__
#endif

PETSC_EXTERN void petscinitializefortran_(int *ierr)
{
  *ierr = PetscInitializeFortran();
}

PETSC_EXTERN void petscsetfortranbasepointers_(char *fnull_character, void *fnull_integer, void *fnull_scalar, void *fnull_double, void *fnull_real, void *fnull_truth, void (*fnull_function)(void), void *fnull_mpi_comm, PETSC_FORTRAN_CHARLEN_T len)
{
  PETSC_NULL_CHARACTER_Fortran = fnull_character;
  PETSC_NULL_INTEGER_Fortran   = fnull_integer;
  PETSC_NULL_SCALAR_Fortran    = fnull_scalar;
  PETSC_NULL_DOUBLE_Fortran    = fnull_double;
  PETSC_NULL_REAL_Fortran      = fnull_real;
  PETSC_NULL_BOOL_Fortran      = fnull_truth;
  PETSC_NULL_FUNCTION_Fortran  = fnull_function;
  PETSC_NULL_MPI_COMM_Fortran  = fnull_mpi_comm;
}

/*
  A valid address for the fortran variable PETSC_NULL_FUNCTION
*/
PETSC_EXTERN void petsc_null_function_(void)
{
  return;
}
