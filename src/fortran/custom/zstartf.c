
#include "src/fortran/custom/zpetsc.h" 
#include "sys.h"

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
extern void petscsetcommonblock_(PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,
                                 int*,int*);
#if defined(__cplusplus)
}
#endif

/*@

    PetscInitializeFortran - Routine that should be called from C after
        the call to PetscInitialize() if one is using a C main program
        that calls Fortran routines that call PETSc routines.

.seealso:  PetscFortranObjectToCObject(), PetscCObjectToFortranObject(),
           PetscInitialize()

.keywords: Mixing C and Fortran, passing PETSc objects to Fortran
@*/

void PetscInitializeFortran(void)
{
  PetscFortranAddr s1,s2,s3;
  int              c1,c2;

  s1 = PetscFromPointer(VIEWER_STDOUT_SELF);
  s2 = PetscFromPointer(VIEWER_STDERR_SELF);
  s3 = PetscFromPointer(VIEWER_STDOUT_WORLD);
  c1 = PetscFromPointerComm(PETSC_COMM_WORLD);
  c2 = PetscFromPointerComm(PETSC_COMM_SELF);
  petscsetcommonblock_(&s1,&s2,&s3,&c1,&c2);
}
  
#if defined(__cplusplus)
extern "C" {
#endif

void petscinitializefortran_(void)
{
  PetscInitializeFortran();
}

#if defined(USES_CPTOFCD)
void petscsetfortranbasepointers_(void *fnull,_fcd fcnull,void *ffnull)
{
  PETSC_NULL_Fortran            = fnull;
  PETSC_NULL_CHARACTER_Fortran  = _fcdtocp(fcnull);
  PETSC_NULL_FUNCTION_Fortran   = ffnull;
}
#else
void petscsetfortranbasepointers_(void *fnull,char *fcnull,void *ffnull)
{
  PETSC_NULL_Fortran            = fnull;
  PETSC_NULL_CHARACTER_Fortran  = fcnull;
  PETSC_NULL_FUNCTION_Fortran   = ffnull;
}
#endif 

#if defined(__cplusplus)
}
#endif
