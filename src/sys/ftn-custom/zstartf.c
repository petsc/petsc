
#include <petsc-private/fortranimpl.h> 

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscinitializefortran_       PETSCINITIALIZEFORTRAN
#define petscsetcommonblock_          PETSCSETCOMMONBLOCK
#define petscsetfortranbasepointers_  PETSCSETFORTRANBASEPOINTERS
#define petsc_null_function_          PETSC_NULL_FUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitializefortran_       petscinitializefortran
#define petscsetcommonblock_          petscsetcommonblock
#define petscsetfortranbasepointers_  petscsetfortranbasepointers
#define petsc_null_function_          petsc_null_function
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define petsc_null_function_  petsc_null_function__
#endif

EXTERN_C_BEGIN
extern void PETSC_STDCALL petscsetcommonblock_(int*,int*);
EXTERN_C_END

/*@C
   PetscInitializeFortran - Routine that should be called soon AFTER
   the call to PetscInitialize() if one is using a C main program
   that calls Fortran routines that in turn call PETSc routines.

   Collective on PETSC_COMM_WORLD

   Level: beginner

   Notes:
   PetscInitializeFortran() initializes some of the default viewers,
   communicators, etc. for use in the Fortran if a user's main program is
   written in C.  PetscInitializeFortran() is NOT needed if a user's main
   program is written in Fortran; in this case, just calling
   PetscInitialize() in the main (Fortran) program is sufficient.

.seealso:  PetscInitialize()

.keywords: Mixing C and Fortran, passing PETSc objects to Fortran
@*/
PetscErrorCode PetscInitializeFortran(void)
{
  PetscMPIInt c1=0,c2=0;

  if (PETSC_COMM_WORLD) {
    c1 =  MPI_Comm_c2f(PETSC_COMM_WORLD);
  }
  /* PETSC_COMM_SELF is defined as MPI_COMM_SELF */
  c2 =  MPI_Comm_c2f(PETSC_COMM_SELF);
  petscsetcommonblock_(&c1,&c2);
  return 0;
}
  
EXTERN_C_BEGIN

void PETSC_STDCALL petscinitializefortran_(int *ierr)
{
  *ierr = PetscInitializeFortran();
}

void PETSC_STDCALL petscsetfortranbasepointers_(char *fnull_character PETSC_MIXED_LEN(len),
                                  void *fnull_integer,void* fnull,void *fnull_scalar,void * fnull_double,
  				  void *fnull_real,void *fnull_object,
                                  void* fnull_truth,void (*fnull_function)(void) PETSC_END_LEN(len))
{
  PETSC_NULL_CHARACTER_Fortran  = fnull_character;
  PETSC_NULL_INTEGER_Fortran    = fnull_integer;
  PETSC_NULL_Fortran            = fnull;
  PETSC_NULL_SCALAR_Fortran     = fnull_scalar;
  PETSC_NULL_DOUBLE_Fortran     = fnull_double;
  PETSC_NULL_REAL_Fortran       = fnull_real;
  PETSC_NULL_OBJECT_Fortran     = fnull_object;
  PETSC_NULL_BOOL_Fortran      = fnull_truth;
  PETSC_NULL_FUNCTION_Fortran   = fnull_function;
}

/*
  A valid address for the fortran variable PETSC_NULL_FUNCTION
*/
void petsc_null_function_(void)
{
  return;
}

EXTERN_C_END

