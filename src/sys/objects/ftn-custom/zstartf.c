#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscsetfortranbasepointers_ PETSCSETFORTRANBASEPOINTERS
  #define petsc_null_function_         PETSC_NULL_FUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscsetfortranbasepointers_ petscsetfortranbasepointers
  #define petsc_null_function_         petsc_null_function
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_null_function_ petsc_null_function__
#endif

PETSC_EXTERN void petscsetfortranbasepointers_(char *fnull_character, void *fnull_integer, void *fnull_scalar, void *fnull_double, void *fnull_real, void *fnull_bool, void *fnull_enum, PetscFortranCallbackFn *fnull_function, void *fnull_mpi_comm, void *fnull_integer_array, void *fnull_scalar_array, void *fnull_real_array, F90Array1d *fnull_integer_pointer, F90Array1d *fnull_scalar_pointer, F90Array1d *fnull_real_pointer, PETSC_FORTRAN_CHARLEN_T len PETSC_F90_2PTR_PROTO(ptrdi) PETSC_F90_2PTR_PROTO(ptrds) PETSC_F90_2PTR_PROTO(ptrdr))
{
  PETSC_NULL_CHARACTER_Fortran       = fnull_character;
  PETSC_NULL_INTEGER_Fortran         = fnull_integer;
  PETSC_NULL_SCALAR_Fortran          = fnull_scalar;
  PETSC_NULL_DOUBLE_Fortran          = fnull_double;
  PETSC_NULL_REAL_Fortran            = fnull_real;
  PETSC_NULL_BOOL_Fortran            = fnull_bool;
  PETSC_NULL_ENUM_Fortran            = fnull_enum;
  PETSC_NULL_FUNCTION_Fortran        = fnull_function;
  PETSC_NULL_MPI_COMM_Fortran        = fnull_mpi_comm;
  PETSC_NULL_INTEGER_ARRAY_Fortran   = fnull_integer_array;
  PETSC_NULL_SCALAR_ARRAY_Fortran    = fnull_scalar_array;
  PETSC_NULL_REAL_ARRAY_Fortran      = fnull_real_array;
  PETSC_NULL_INTEGER_POINTER_Fortran = (void *)fnull_integer_pointer;
  PETSC_NULL_SCALAR_POINTER_Fortran  = (void *)fnull_scalar_pointer;
  PETSC_NULL_REAL_POINTER_Fortran    = (void *)fnull_real_pointer;
}

/*
  A valid address for the fortran variable PETSC_NULL_FUNCTION
*/
PETSC_EXTERN void petsc_null_function_(void)
{
  return;
}
