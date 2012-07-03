#include <petsc-private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define ksppythonsettype_            KSPPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ksppythonsettype_            ksppythonsettype
#endif


EXTERN_C_BEGIN

void PETSC_STDCALL  ksppythonsettype_(KSP *ksp, CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = KSPPythonSetType(*ksp,t);
  FREECHAR(name,t);
}


EXTERN_C_END
