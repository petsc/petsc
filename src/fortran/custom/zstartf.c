
#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"
#include <stdio.h>

#ifdef HAVE_FORTRAN_CAPS
#define petscinitializefortran_       PETSCINITIALIZEFORTRAN
#define petscsetcommonblock_          PETSCSETCOMMONBLOCK
#define petscsetfortranbasepointers_  PETSCSETFORTRANBASEPOINTERS
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define petscinitializefortran_       petscinitializefortran
#define petscsetcommonblock_          petscsetcommonblock
#define petscsetfortranbasepointers_  petscsetfortranbasepointers
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void petscsetcommonblock_(int*,int*,int*,int*);
#if defined(__cplusplus)
}
#endif

/*
  This function should be called to be able to use PETSc routines
  from the FORTRAN subroutines, when the main() routine is in C
*/

void PetscInitializeFortran()
{
  int s1,s2,s3,s4;
  s1 = MPIR_FromPointer(VIEWER_STDOUT_SELF);
  s2 = MPIR_FromPointer(VIEWER_STDERR_SELF);
  s3 = MPIR_FromPointer(VIEWER_STDOUT_WORLD);
  s4 = MPIR_FromPointer_Comm(PETSC_COMM_WORLD);
  petscsetcommonblock_(&s1,&s2,&s3,&s4);
}
  
#if defined(__cplusplus)
extern "C" {
#endif

void petscinitializefortran_()
{
  PetscInitializeFortran();
}

#if defined(USES_CPTOFCD)
void petscsetfortranbasepointers_(void *fnull,_fcd fcnull)
{
  PETSC_NULL_Fortran       = fnull;
  PETSC_NULL_CHARACTER_Fortran  = _fcdtocp(fcnull);
}
#else
void petscsetfortranbasepointers_(void *fnull,char *fcnull)
{
  PETSC_NULL_Fortran       = fnull;
  PETSC_NULL_CHARACTER_Fortran  = fcnull;
}
#endif 

#if defined(__cplusplus)
}
#endif
