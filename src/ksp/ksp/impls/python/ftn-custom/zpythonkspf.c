#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define ksppythonsettype_            KSPPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ksppythonsettype_            ksppythonsettype
#endif

PETSC_EXTERN void ksppythonsettype_(KSP *ksp, char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = KSPPythonSetType(*ksp,t);if (*ierr) return;
  FREECHAR(name,t);
}

